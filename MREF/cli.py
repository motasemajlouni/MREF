import os
import argparse
from statistics import fmean
from .mref_metric import MREF, AxisWeights


def _read_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def main():
    parser = argparse.ArgumentParser("Calculate MREF")

    parser.add_argument(
        "-s",
        "--saxis-only",
        dest="saxis_only",
        action="store_true",
        help="Return Saxis only",
    )

    parser.add_argument(
        "-m",
        "--maxis-only",
        dest="maxis_only",
        action="store_true",
        help="Return Maxis only",
    )

    parser.add_argument(
        "-g",
        "--gaxis-only",
        dest="gaxis_only",
        action="store_true",
        help="Return Gaxis only",
    )

    parser.add_argument(
        "-e",
        "--mref-only",
        dest="mref_only",
        action="store_true",
        help="Return MREFscore only",
    )

    parser.add_argument(
        "-r",
        "--ref",
        type=str,
        required=True,
        help="Reference / complex text file path or a string",
    )

    parser.add_argument(
        "-c",
        "--cand",
        type=str,
        required=True,
        help="Candidate / simplified text file path or a string",
    )

    parser.add_argument(
        "-o",
        "--outs",
        type=str,
        default="MREFscore.res",
        help="Output results file path (default: MREFscore.res)",
    )

    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="en",
        help="Language code for MREF (default: en)",
    )

    parser.add_argument(
        "--max-tatoeba-sentences",
        dest="max_tatoeba_sentences",
        type=int,
        default=None,
        help="Optional limit for auto-building POS prior from Tatoeba",
    )

    parser.add_argument(
        "--w-g",
        dest="w_g",
        type=float,
        default=None,
        help="Optional grammaticality weight",
    )

    parser.add_argument(
        "--w-m",
        dest="w_m",
        type=float,
        default=None,
        help="Optional meaning-preservation weight",
    )

    parser.add_argument(
        "--w-s",
        dest="w_s",
        type=float,
        default=None,
        help="Optional simplicity weight",
    )

    args = parser.parse_args()

    weight_args = [args.w_g, args.w_m, args.w_s]

    if any(w is not None for w in weight_args):
        if not all(w is not None for w in weight_args):
            parser.error(
                "If you pass custom weights, you must provide --w-g, --w-m, and --w-s together."
            )

        if any(w < 0 for w in weight_args):
            parser.error("Custom weights must be non-negative.")

        if (args.w_g + args.w_m + args.w_s) <= 0:
            parser.error("At least one custom weight must be positive.")

        axis_weights = AxisWeights(
            w_G=args.w_g,
            w_M=args.w_m,
            w_S=args.w_s,
        )
    else:
        axis_weights = None

    metric = MREF(
        language=args.language,
        max_tatoeba_sentences=args.max_tatoeba_sentences,
    )

    ref_is_file = os.path.isfile(args.ref)
    cand_is_file = os.path.isfile(args.cand)

    # ------------------------------------------------------------------
    # Corpus mode: both inputs are files
    # ------------------------------------------------------------------
    if ref_is_file:
        comps = _read_lines(args.ref)

        if cand_is_file:
            simps = _read_lines(args.cand)

            if len(comps) != len(simps):
                print(
                    f"# of simplified texts in ({args.cand}) does not match "
                    f"the # of complex texts in ({args.ref})"
                )
                return 1

            s_scores = []
            m_scores = []
            g_scores = []
            mref_scores = []
            results = []

            for comp, simp in zip(comps, simps):
                out = metric.score_axes(comp, simp, axis_weights=axis_weights)
                s_scores.append(out["S_axis"])
                m_scores.append(out["M_axis"])
                g_scores.append(out["G_axis"])
                mref_scores.append(out["MREFscore"])

            if args.saxis_only:
                print(f"corpus Saxis = {fmean(s_scores):.6f}")
                for s in s_scores:
                    results.append(f"{s:.6f}\n")
                with open(args.outs, "w", encoding="utf-8") as out_file:
                    out_file.write("S_axis\n")
                    out_file.writelines(results)

            elif args.maxis_only:
                print(f"corpus Maxis = {fmean(m_scores):.6f}")
                for m in m_scores:
                    results.append(f"{m:.6f}\n")
                with open(args.outs, "w", encoding="utf-8") as out_file:
                    out_file.write("M_axis\n")
                    out_file.writelines(results)

            elif args.gaxis_only:
                print(f"corpus Gaxis = {fmean(g_scores):.6f}")
                for g in g_scores:
                    results.append(f"{g:.6f}\n")
                with open(args.outs, "w", encoding="utf-8") as out_file:
                    out_file.write("G_axis\n")
                    out_file.writelines(results)

            elif args.mref_only:
                print(f"corpus MREFscore = {fmean(mref_scores):.6f}")
                for ms in mref_scores:
                    results.append(f"{ms:.6f}\n")
                with open(args.outs, "w", encoding="utf-8") as out_file:
                    out_file.write("MREFscore\n")
                    out_file.writelines(results)

            else:
                print(f"corpus Saxis = {fmean(s_scores):.6f}")
                print(f"corpus Maxis = {fmean(m_scores):.6f}")
                print(f"corpus Gaxis = {fmean(g_scores):.6f}")
                print(f"corpus MREFscore = {fmean(mref_scores):.6f}")

                for s, m, g, ms in zip(s_scores, m_scores, g_scores, mref_scores):
                    results.append(f"{s:.6f} ; {m:.6f} ; {g:.6f} ; {ms:.6f}\n")

                with open(args.outs, "w", encoding="utf-8") as out_file:
                    out_file.write("S_axis ; M_axis ; G_axis ; MREFscore\n")
                    out_file.writelines(results)

        else:
            print(f"simplified texts file ({args.cand}) does not exist")
            return 1

    elif cand_is_file:
        print(f"complex texts file ({args.ref}) does not exist")
        return 1

    # ------------------------------------------------------------------
    # Single-pair mode: both inputs are strings
    # ------------------------------------------------------------------
    else:
        comp = args.ref
        simp = args.cand
        out = metric.score_axes(comp, simp, axis_weights=axis_weights)

        s = out["S_axis"]
        m = out["M_axis"]
        g = out["G_axis"]
        ms = out["MREFscore"]

        if args.saxis_only:
            print(f"Saxis = {s:.6f}")

        elif args.maxis_only:
            print(f"Maxis = {m:.6f}")

        elif args.gaxis_only:
            print(f"Gaxis = {g:.6f}")

        elif args.mref_only:
            print(f"MREFscore = {ms:.6f}")

        else:
            print(f"Saxis = {s:.6f}")
            print(f"Maxis = {m:.6f}")
            print(f"Gaxis = {g:.6f}")
            print(f"MREFscore = {ms:.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
