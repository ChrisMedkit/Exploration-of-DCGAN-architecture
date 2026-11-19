"""Score every finished run on the same held-out test captions and write the results table (report §3.2).

    python evaluate_all.py            # -> reports/results.csv, reports/figures/fid_clip_by_experiment.png

TODO (Plan steps 10-11):
  1. cache = data.load_image_cache(); rows = data.class_split(cache["labels"])["test"]
     real = data.to_model_range(cache["images"][rows])
     test_t = text_encoder.load_text_cache(cache["ids"])[rows, 0].float()   # one caption per test image
  2. for each models/<experiment>_s<seed>/last.pt:
       G = model.build_models(conditional)[0]; G.load_state_dict(ck["G"])
       fake = evaluate.generate_for_eval(G, test_t if conditional else None, seed=seed)
       row = dict(experiment, seed, fid=evaluate.compute_fid(real, fake),
                  clip_score=evaluate.clip_score(fake, test_t[:len(fake)]) if conditional else None)
  3. also score the reference rows: real images + true captions (upper bound), real images + shuffled captions (chance)
  4. pd.DataFrame(rows).to_csv(config.ROOT / "reports" / "results.csv", index=False)
  5. evaluate.summarize_seeds(df, "fid"), summarize_seeds(df, "clip_score") -> print + save
  6. bar chart with error bars (mean +- std over seeds) per experiment for both metrics
"""


def main() -> None:
    raise NotImplementedError("Milestone 3 (Plan steps 10-11)")


if __name__ == "__main__":
    main()
