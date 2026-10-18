"""CPU smoke test for the whole HAIKU ZERO pipeline. No framework, just asserts.

    python test_smoke.py

Runs in well under a minute on CPU and needs only torch + tokenizers (the API check
binds a throwaway port and exercises backend.server against the tiny checkpoint).
"""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import threading
import urllib.request
from pathlib import Path

import torch

import backend.server as server_mod
import backend.train as train_mod
from backend import generate, meter_check, prepare_data, train_tokenizer
from backend.config import Config
from backend.dataset import build_loaders, encode_records, pack, split_records
from backend.model import HaikuZeroLM

WORDS = "autumn moonlight pond frog snow crow wind rain leaf river stone quiet dusk bird".split()


def poem_words(seed: int, count: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(count))


def make_fixture(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    rows = [["", "0", "1", "2", "source", "hash"]]
    good = [[poem_words(i * 7 + 1, 3), poem_words(i * 7 + 2, 4), poem_words(i * 7 + 3, 3)]
            for i in range(40)]
    for i, lines in enumerate(good):
        rows.append([str(i), *lines, "fixture", f"HASH{i}"])
    rows.append(["900", *good[1], "fixture", "DUPE"])                     # duplicate -> dropped
    rows.append(["902", good[2][0], "x" * 70, good[2][2], "fixture", "LONG"])
    rows.append(["903", good[3][0], "see https://example.com", good[3][2], "fixture", "URL"])
    weird = "\u00fc\u00fc\u00fc\u00fc\u00fc\u00fc x"          # letters present, >50% non-ascii
    rows.append(["904", weird, weird, weird, "fixture", "NONASCII"])
    with (raw / "fixture.csv").open("w", encoding="utf-8", newline="") as f:
        for row in rows:
            f.write(",".join(row) + "\n")
    assert len({tuple(g) for g in good}) == 40, "fixture is not unique"


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="haiku_smoke_"))
    try:
        raw = tmp / "raw"
        make_fixture(raw)

        # --- Module 1: cleaning ---
        clean, meter, tok_path, stats_path = (tmp / "clean.txt", tmp / "meter.txt",
                                              tmp / "tokenizer.json", tmp / "stats.json")
        stats = prepare_data.main(["--raw-dir", str(raw), "--out", str(clean),
                                   "--meter-out", str(meter), "--stats", str(stats_path)])
        assert stats["kept"] == 40, stats
        assert stats["rejected"] == {"no_letters": 1, "duplicate": 1, "line_too_long": 1,
                                     "url_or_mention": 1, "too_much_non_ascii": 1}, stats
        text = clean.read_text(encoding="utf-8")
        assert text.count("<eos>\n") == 40
        assert meter.exists() and json.loads(stats_path.read_text(encoding="utf-8"))["kept"] == 40
        print(f"[1] prepare_data ok: 40 kept, rejected {stats['rejected']}, meter subset {stats['meter_subset']}")

        # --- Module 2: tokenizer ---
        tok = train_tokenizer.train_tokenizer(clean, tok_path, vocab_size=256, show=False)
        info = train_tokenizer.sanity(tok, train_tokenizer.read_records(clean), show=False)
        assert info["roundtrip_ok"], info
        assert info["special_ids"] == {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<seed>": 3}, info
        assert info["eos_is_single_token"]
        assert info["byte_lossless"], info    # unseen bytes must survive encode->decode
        print(f"[2] tokenizer ok: vocab {info['real_vocab_size']}, "
              f"{info['avg_tokens_per_haiku']} tokens/haiku")

        # --- Module 3: split + packing ---
        records = train_tokenizer.read_records(clean)
        train_recs, val_recs = split_records(records, 0.03, 1337)
        assert not set(train_recs) & set(val_recs), "train/val leak"
        bos_id, eos_id = int(tok.token_to_id("<bos>")), int(tok.token_to_id("<eos>"))
        chunks, masks = encode_records(records, tok)
        assert all(c[0] == bos_id and c[-1] == eos_id for c in chunks), "every poem needs <bos> ... <eos>"
        x, y = pack(chunks, 64)
        assert x.shape == y.shape and x.shape[1] == 64
        assert torch.equal(x[:, 1:], y[:, :-1]), "windows are not shifted by one"
        assert eos_id in y and bos_id in x
        # prefix augmentation: given tokens become -100 targets, and val data stays unmasked
        masked_chunks, masked = encode_records(records, tok, prefix_mask_prob=1.0, rng=random.Random(7))
        assert any(any(row) for row in masked), "prefix masking produced nothing"
        _, y_masked = pack(masked_chunks, 64, masked)
        assert (y_masked == -100).any() and (y_masked != -100).any(dim=1).all()
        assert (y != -100).all(), "unmasked packing must not drop targets"
        # regression: a masked prefix must never swallow a whole window. Records longer
        # than seq_length (the poetry pretraining corpus produces them; haiku never did)
        # could make `given` reach the window size, leaving a window with no target at
        # all and pack() refusing to build the dataset.
        long_rec = " ".join(["wandering"] * 40)
        long_chunk = encode_records([long_rec], tok)[0][0]
        assert len(long_chunk) > 66, f"fixture must exceed one window, got {len(long_chunk)}"
        full_window_mask = [False] * len(long_chunk)
        for i in range(1, 65):                      # a run exactly as long as the window
            full_window_mask[i] = True
        try:
            pack([long_chunk], 64, [full_window_mask])
            guard_fired = False
        except SystemExit:
            guard_fired = True
        assert guard_fired, "pack must refuse a window with no targets (the guard is load-bearing)"
        capped_chunks, capped_masks = encode_records([long_rec], tok, prefix_mask_prob=1.0,
                                                     rng=random.Random(1), max_mask=62)
        assert max(sum(row) for row in capped_masks) <= 62, "masked run exceeded the window cap"
        _, y_capped = pack(capped_chunks, 64, capped_masks)      # must not raise
        assert (y_capped != -100).any(dim=1).all(), "a window lost every target"
        # seed conditioning: [<bos> <seed> word \n poem <eos>], seed region is given context
        seed_tok = int(tok.token_to_id("<seed>"))
        s_chunks, s_masks = encode_records(records, tok, seed_prob=1.0, rng=random.Random(3))
        assert all(c[0] == bos_id and c[1] == seed_tok and c[-1] == eos_id for c in s_chunks)
        assert all(int(tok.encode("\n").ids[0]) in c for c in s_chunks), "seed line needs a newline"
        assert all(m[0] and m[1] and not m[-1] for m in s_masks), "seed must be given context"
        ctx = generate.seed_context(tok, "snow")
        assert ctx[0] == bos_id and ctx[1] == seed_tok and ctx[-1] == int(tok.encode("\n").ids[0])
        assert tok.decode(ctx[2:-1]) == " snow", "training and inference seed formats must match"
        print(f"[3] dataset ok: {x.shape[0]} windows, {x.numel()} tokens, "
              f"{int((y_masked == -100).sum())} masked targets in the augmented copy")

        # --- Modules 4-7: model ---
        cfg = Config(vocab_size=info["real_vocab_size"], embed_dim=32, num_heads=4,
                     num_layers=2, ffn_dim=64, dropout=0.0, seq_length=64)
        model = HaikuZeroLM(cfg)
        logits, loss = model(x[:2], y[:2])
        assert logits.shape == (2, 64, cfg.vocab_size), logits.shape
        assert logits.dtype == torch.float32, logits.dtype
        assert torch.isfinite(loss), loss
        assert model.lm_head.weight is model.emb.token.weight, "weights not tied"
        flipped = x[:1].clone()
        flipped[0, -1] = (flipped[0, -1] + 7) % cfg.vocab_size
        model.eval()
        a, _ = model(x[:1])
        b, _ = model(flipped)
        assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6), "attention is not causal"
        print(f"[4] model ok: {model.num_params() / 1e6:.3f}M params, causal, float32 head")

        # --- Module 9: generation shapes (untrained model) ---
        lines, _, gen_ids = generate.sample_haiku(model, tok, "autumn", temperature=1.0,
                                                 max_new_tokens=12, device="cpu", amp=False, seed=0)
        assert 1 <= len(lines) <= 3 and gen_ids, lines
        assert lines[0].lower().startswith("autumn"), lines       # plain mode honours seed_first too
        assert not any(generate.has_junk(l) for l in lines), lines  # and the static junk mask
        free_lines, _, _ = generate.sample_haiku(model, tok, "autumn", temperature=1.0,
                                                max_new_tokens=12, device="cpu", amp=False,
                                                seed=0, seed_first=False)
        assert not free_lines[0].lower().startswith("autumn"), free_lines
        assert len(generate.junk_token_ids(tok)) > 0, "the junk token mask is empty"
        print(f"[5] generate ok: {lines}")

        # --- Module 10 + metered decoding ---
        # candidates are FULL line texts now (line so far + candidate), scored on the decoded text
        assert generate.allowed_candidates(["the old silent pond"], "the old silent", 5) == [0]
        assert generate.allowed_candidates(["the old silent pond\n"], "the old silent", 5) == [0]
        assert generate.allowed_candidates(["the old silent pond\n"], "the old silent", 4) == []
        assert generate.allowed_candidates(["a frog jumps into the pond"], "a frog", 5) == []  # overshoot
        assert generate.allowed_candidates(["the old silent pond"], "the old silent pond", 5) == []
        # no zero-syllable junk: this is the regression test for the "?????" / ". . . ." spam
        assert generate.has_junk("soft night?????the")
        assert generate.has_junk("early morning . . .")
        assert not generate.has_junk("midnight rain and it's the freeze")
        assert generate.allowed_candidates(["the old silent pond?"], "the old", 5) == []
        assert generate.allowed_candidates(["the old silent pond. . ."], "the old", 5) == []
        assert generate.allowed_candidates(["naïve café light"], "naïve", 5) == [0]   # accented letters are fine
        nl_id = tok.encode("\n").ids[0]
        assert "\n" in tok.decode([nl_id]), "newline is not one token"
        ml, mpat, exact = generate.sample_haiku_metered(model, tok, "", targets=(3, 4, 3),
                                                       max_new_tokens=120, device="cpu", amp=False, seed=2)
        assert len(ml) == 3 and len(mpat) == 3, (ml, mpat)      # strict mode always yields 3 lines
        assert all(p <= t for p, t in zip(mpat, (3, 4, 3))), (ml, mpat)   # never overshoots
        print(f"[6b] metered decoding ok: {mpat} (exact {exact})")
        bo_lines, bo_pat, bo_exact = generate.best_of_haiku(model, tok, "snow", n=3, strict=True,
                                                          max_new_tokens=200, device="cpu",
                                                          amp=False, seed=5)
        assert len(bo_lines) == 3, bo_lines            # best-of must keep strict's 3-line promise
        assert generate.score_text(model, tok, [bos_id], "autumn moonlight") < 0, "score must be negative"
        assert generate.seed_adherence(["snow falls", "quiet", "still"], "snow") == 1.0
        assert generate.seed_adherence(["autumnal light", "x", "y"], "autumn") == 1.0
        assert generate.seed_adherence(["breezes", "x", "y"], "breeze") == 1.0
        assert generate.seed_adherence(["summer heat", "x", "y"], "autumn") == 0.0
        assert generate.seed_adherence(["anything at all"], "") == 0.0
        assert generate.seed_adherence(["the old pond", "x", "y"], "old pond") == 1.0   # multi-word
        # non-word rerank: real words score, invented ones do not (inflections allowed)
        words = frozenset({"autumn", "moonlight", "pond", "frog", "snow"})
        assert generate.attested_fraction(["autumn moonlight"], words) == 1.0
        assert generate.attested_fraction(["snows fall"], words) == 0.5       # 'fall' unseen, 'snows'->snow
        assert generate.attested_fraction(["windshavel cournament"], words) == 0.0
        # seed_first writes the seed as the poem's first word
        primed, _, _ = generate.sample_haiku_metered(model, tok, "snow", targets=(3, 4, 3),
                                                   max_new_tokens=120, device="cpu", amp=False,
                                                   seed=4, seed_first=True)
        assert primed and primed[0].lower().startswith("snow"), primed
        unprimed, _, _ = generate.sample_haiku_metered(model, tok, "snow", targets=(3, 4, 3),
                                                     max_new_tokens=120, device="cpu", amp=False,
                                                     seed=4, seed_first=False)
        assert unprimed and not unprimed[0].lower().startswith("snow"), unprimed
        assert generate.has_junk("winter night------Suno")   # dash run
        assert generate.has_junk("winter night--")            # trailing dash
        assert not generate.has_junk("winter-night light")     # a joined hyphen is fine
        # anti-stutter: seeding invites "autumn autumn day"
        assert generate.has_stutter("autumn autumn day")
        assert generate.has_stutter("the the pond")
        assert not generate.has_stutter("the old pond")
        assert generate.allowed_candidates(["autumn autumn day"], "autumn", 5) == []

        assert meter_check.line_syllables("the old silent pond") == 5
        assert meter_check.line_syllables("a frog jumps into the pond") == 7
        assert meter_check.is_haiku(["the old silent pond", "a frog jumps into the pond",
                                     "splash silence again"])
        assert meter_check.distance((5, 7, 5)) == 0 and meter_check.distance((4, 6, 6)) == 3
        # seeded prompt must be cleaned: junk used to be pasted in as the poem's first line
        assert generate.clean_seed("snow!") == "snow"
        assert generate.clean_seed("SNOW") == "snow"
        assert generate.clean_seed("snow3") == "snow"
        assert generate.clean_seed("\U0001F338") == ""
        assert generate.clean_seed("  blood-red  ") == "blood red"   # the hyphen is not a word
        # repeat check must look at the candidate's NEW text only, or every continuation of a
        # line that already used a content word looked like a repeat and the filter gave up
        assert generate.adds_repeat_now(" blood", {"blood"})
        assert generate.adds_repeat_now(" blood's", {"blood"})       # stem, so possessive counts
        assert not generate.adds_repeat_now(" of", {"blood"})
        assert not generate.adds_repeat_now("", {"blood"})
        assert generate.candidate_weight(" blood of blood", " blood", {"blood"}, None, 1.0) == 0.0
        assert generate.candidate_weight(" blood of iron", " blood", {"blood"}, None, 1.0) > 0
        # a fused word is dead weight too ("blood"+"y" = bloody), which is why the seed no
        # longer grows into a different word
        words = frozenset({"blood", "river", "stream"})
        assert generate.candidate_weight(" bloody", " blood", set(), words, 1.0) == 0.0
        assert generate.candidate_weight(" river stream", " river", set(), words, 1.0) > 0
        # top-p can leave ONE candidate (p=1, tail 0). The tail needs weight, or zeroing the
        # repeat leaves nothing to sample and the draw falls back to the repeat itself.
        nucleus = torch.tensor([1.0, 0.0, 0.0])
        full = torch.tensor([0.99, 0.05, 0.01])
        weights = generate.tail_weighted(nucleus, full, 3)
        assert weights[0] == 1.0 and weights[1] > 0 and weights[2] > 0, weights
        # a quote mark floating between words is a seam leftover, a possessive apostrophe is not
        assert generate.has_junk("suits for a time inde \u2019")
        assert generate.has_junk("a haiku of ' hello")
        assert generate.has_junk("the string's boughs'")     # apostrophe left hanging at the end
        assert not generate.has_junk("the birds' song")
        assert not generate.has_junk("it's a bird")
        assert generate.has_stutter("Tell'st'st'stwas so")          # fragment repeat at a seam
        assert not generate.has_stutter("mama said")

        assert generate.repeated_content_words(["blood and blood are all"])
        assert generate.repeated_content_words(["snow's in the snow"])   # stem, not spelling
        assert not generate.repeated_content_words(["the old pond", "a frog jumps in"])
        assert generate._stem("snow's") == generate._stem("snows") == "snow"
        # word seam: "river"+"stream" must lose to " river stream" (the fused token is the bug)
        assert generate.keeps_word_boundary("river stream", "river", None)
        assert not generate.keeps_word_boundary("riverstream", "river", frozenset({"river"}))
        assert generate.keeps_word_boundary("riverstream", "rivers", frozenset({"river"}))
        # invented words: the corpora are the vocabulary, inflections allowed. A candidate that
        # FINISHES one is dead weight, so "shavement" is never preferred over a real word.
        vocab = frozenset({"snow", "moon", "moonlight"})
        assert generate.known_word("snows", vocab) and generate.known_word("moonlight", vocab)
        assert not generate.known_word("shavement", vocab)
        assert generate.known_word("boughs'", frozenset({"boughs"}))   # a possessive is not a word
        assert generate.added_words(" snow falls ", " snow") == ["falls"]
        assert generate.added_words(" moonl", " moon") == []        # still being written
        assert generate.added_words(" shavement\n", " snow") == ["shavement"]   # line end counts
        assert generate.adds_invented_word(" shavement ", " snow", vocab)
        assert not generate.adds_invented_word(" moonlight", " snow", vocab)
        assert generate.candidate_weight(" shavement ", " snow", set(), vocab, 1.0) == 0.0
        assert generate.candidate_weight(" moonlight ", " snow", set(), vocab, 1.0) > 0
        # free meter is bounded: a line may close anywhere, but it may not run on forever
        assert generate.allowed_candidates(["the old silent pond"], "the old silent", None,
                                           frozenset(), None, max_syllables=5) == [0]
        assert generate.allowed_candidates(["the old silent pond in rain"],
                                           "the old silent pond in", None, frozenset(), None,
                                           max_syllables=5) == []
        assert generate.allowed_candidates(["the old silent pond in\n"],
                                           "the old silent pond in", None, frozenset(), None,
                                           max_syllables=5) == [0]
        for line in free_lines:
            assert meter_check.line_syllables(line) <= generate.FREE_MAX_SYLLABLES, free_lines
        print("[6] meter ok")

        # --- repetition: ranking prefers a poem that spends no word twice ---
        real_sample, real_score = generate.sample_haiku, generate.score_text
        draws = []

        def fake_sample(*_a, **_kw):
            draws.append(1)
            return (["blood and blood are all"] if len(draws) == 1
                    else ["a stone in the rain"]), None, None

        try:
            generate.sample_haiku = fake_sample
            generate.score_text = lambda *_a, **_kw: 0.0
            lines, _pat, _exact = generate.best_of_haiku(model, tok, "stone", n=2,
                                                        device="cpu", amp=False, seed=1)
        finally:
            generate.sample_haiku, generate.score_text = real_sample, real_score
        assert lines == ["a stone in the rain"], lines
        print("[6e] repetition ranking ok")

        # --- content filter: banned words are rejected at sampling time (no retrain) ---
        old_banned = generate.BANNED_PATH
        try:
            (tmp / "banned.txt").write_text("# test list\nchurch\njesus\n# cross is allowed\n",
                                            encoding="utf-8")
            generate.BANNED_PATH = tmp / "banned.txt"
            generate.clear_word_caches()
            assert generate.has_banned("a church of stone")
            assert not generate.has_banned("churchgoers sing")      # whole words only
            assert not generate.has_banned("the cross stands")      # commented out = allowed
            assert not generate.has_banned("church", exempt=frozenset({"church"}))
            assert generate.blocked_words("jesus") == frozenset({"church"})   # seed exempt
            assert generate.blocked_words("snow") == frozenset({"church", "jesus"})
            # editing the list must take effect without a restart (no cache_clear here)
            (tmp / "banned.txt").write_text("church\nsnowman\n", encoding="utf-8")
            assert "snowman" in generate.banned_words()

            # rejection sampling: a dirty first draw must be dropped, not ranked
            real_sample, real_score = generate.sample_haiku, generate.score_text
            draws = []

            def fake_sample(*_a, **_kw):
                draws.append(1)
                return (["church bells in snow"] if len(draws) == 1
                        else ["autumn moonlight"]), None, None

            try:
                generate.sample_haiku = fake_sample
                generate.score_text = lambda *_a, **_kw: 0.0
                lines, _pat, _exact = generate.best_of_haiku(model, tok, "snow", n=2,
                                                             device="cpu", amp=False, seed=1)
            finally:
                generate.sample_haiku, generate.score_text = real_sample, real_score
            assert lines == ["autumn moonlight"], lines
            assert len(draws) >= 2, draws
            print("[6d] banned-word filter ok")
        finally:
            generate.BANNED_PATH = old_banned
            generate.clear_word_caches()

        # --- wordlist cache: a stale cache must never be mistaken for the real one ---
        old_corpora, old_path, old_line = (generate.WORDLIST_CORPORA, generate.WORDLIST_PATH,
                                           generate.SOURCES_LINE)
        try:
            (tmp / "wl_c1.txt").write_text("autumn moonlight pond", encoding="utf-8")
            (tmp / "wl_c2.txt").write_text("frog leaps", encoding="utf-8")
            generate.WORDLIST_PATH = tmp / "wl_cache.txt"
            generate.WORDLIST_CORPORA = (tmp / "wl_c1.txt",)
            generate.SOURCES_LINE = "# sources: wl_c1.txt"
            generate.load_wordlist.cache_clear()
            first = generate.load_wordlist()
            assert "autumn" in first and "frog" not in first, first
            generate.WORDLIST_CORPORA = (tmp / "wl_c1.txt", tmp / "wl_c2.txt")
            generate.SOURCES_LINE = "# sources: wl_c1.txt wl_c2.txt"
            generate.load_wordlist.cache_clear()
            second = generate.load_wordlist()   # the cached file names one corpus only
            assert "frog" in second and "autumn" in second, second
            assert generate.load_wordlist.cache_clear() is None
            print("[6c] wordlist cache invalidation ok")
        finally:
            generate.WORDLIST_CORPORA, generate.WORDLIST_PATH = old_corpora, old_path
            generate.SOURCES_LINE = old_line
            generate.load_wordlist.cache_clear()

        # --- Module 8: two real training steps + checkpoint round trip ---
        tiny = Config(**{**cfg.to_dict(), "batch_size": 2, "epochs": 1, "val_fraction": 0.2})
        tr, va, tok2, ds = build_loaders(tiny, corpus=clean, tokenizer_path=tok_path,
                                         batch_size=2, verbose=False)
        assert ds["steps_per_epoch"] >= 1 and len(va.dataset) >= 1, ds
        ckpt = tmp / "ckpt"
        best = train_mod.train_run(tiny, tr, va, tok2, ckpt_dir=ckpt, epochs=1, max_steps=2,
                                   device="cpu", amp=False, preview=True, log_path=tmp / "log.csv",
                                   tokenizer_path=tok_path, verbose=False)
        assert torch.isfinite(torch.tensor(best)), best
        assert (ckpt / "last.pt").exists() and (ckpt / "best.pt").exists()
        assert (tmp / "log.csv").read_text(encoding="utf-8").count("\n") >= 2
        reloaded, tok3, _, _, payload = generate.load_model(ckpt / "best.pt", tok_path, "cpu")
        lines3, _, _ = generate.sample_haiku(reloaded, tok3, "snow", max_new_tokens=6,
                                             device="cpu", amp=False, seed=1)
        assert lines3 and payload["step"] == 2, payload
        print(f"[7] train + checkpoint ok: val {best:.3f}, reloaded at step {payload['step']}")

        # --- Module 12: the HTTP API the front end talks to (real socket, tiny checkpoint) ---
        assert server_mod.clamped({"count": 99}, "count", 1, 1, server_mod.MAX_COUNT) == server_mod.MAX_COUNT
        assert server_mod.clamped({"temperature": "hot"}, "temperature", 0.8, 0.1, 2.0) == 0.8
        assert server_mod.registry_payload()[0]["id"], "real checkpoints must be listed"
        real_registry = server_mod.model_registry
        server_mod.model_registry = lambda: {"smoke": ckpt / "best.pt"}
        server_mod.Handler.log_message = lambda *a, **k: None            # keep the test output clean
        httpd = server_mod.ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            assert json.loads(urllib.request.urlopen(f"{base}/api/health", timeout=30).read())["ok"]
            listed = json.loads(urllib.request.urlopen(f"{base}/api/models", timeout=30).read())
            assert [m["id"] for m in listed["models"]] == ["smoke"] and listed["default"] == "smoke"
            body = json.dumps({"prompt": "snow", "model": "smoke", "count": 1, "strict": True,
                               "best_of": 1, "max_new_tokens": 60, "seed": 3}).encode()
            request = urllib.request.Request(f"{base}/api/generate", data=body,
                                             headers={"Content-Type": "application/json"})
            result = json.loads(urllib.request.urlopen(request, timeout=180).read())
            assert result["poems"] and result["poems"][0]["lines"], result
            assert set(result["poems"][0]) >= {"lines", "pattern", "exact", "logprob", "repeated_rate"}
            assert result["seconds"] >= 0 and result["best_of"] == 1
        finally:
            httpd.shutdown()
            httpd.server_close()
            server_mod.model_registry = real_registry
        print(f"[8] api ok: /api/health, /api/models, /api/generate -> {result['poems'][0]['lines']}")

        print("\nALL SMOKE CHECKS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
