"""Taken from ERRANT
- https://github.com/chrisjbryant/errant/blob/main/errant/commands/compare_m2.py
- https://github.com/chrisjbryant/errant/blob/main/errant/commands/parallel_to_m2.py

Arguments of the functions are modified for the ease of use in the current project.
Some code are refactored into functions for usability.
"""
from collections import Counter

import errant
from errant.commands.compare_m2 import (
    compareEdits,
    computeFScore,
    merge_dict,
    print_table,
    processCategories,
    simplify_edits,
)
from errant.commands.parallel_to_m2 import noop_edit


def process_edits(edits, evaluation_type, single, multi, filt):
    coder_dict = {}
    # Add an explicit noop edit if there are no edits.
    if not edits:
        edits = [[-1, -1, "noop", "-NONE-", 0]]
    # Loop through the edits
    for edit in edits:
        # Name the edit elements for clarity
        start = edit[0]
        end = edit[1]
        cat = edit[2]
        cor = edit[3]
        coder = edit[4]
        # Add the coder to the coder_dict if necessary
        if coder not in coder_dict:
            coder_dict[coder] = {}

        # Optionally apply filters based on args
        # 1. UNK type edits are only useful for detection, not correction.
        if evaluation_type != "dt" and evaluation_type != "ds" and cat == "UNK":
            continue
        # 2. Only evaluate single token edits; i.e. 0:1, 1:0 or 1:1
        if single and (end - start >= 2 or len(cor.split()) >= 2):
            continue
        # 3. Only evaluate multi token edits; i.e. 2+:n or n:2+
        if multi and end - start < 2 and len(cor.split()) < 2:
            continue
        # 4. If there is a filter, ignore the specified error types
        if filt and cat in filt:
            continue

        # Token Based Detection
        if evaluation_type == "dt":
            # Preserve noop edits.
            if start == -1:
                if (start, start) in coder_dict[coder].keys():
                    coder_dict[coder][(start, start)].append(cat)
                else:
                    coder_dict[coder][(start, start)] = [cat]
            # Insertions defined as affecting the token on the right
            elif start == end and start >= 0:
                if (start, start + 1) in coder_dict[coder].keys():
                    coder_dict[coder][(start, start + 1)].append(cat)
                else:
                    coder_dict[coder][(start, start + 1)] = [cat]
            # Edit spans are split for each token in the range.
            else:
                for tok_id in range(start, end):
                    if (tok_id, tok_id + 1) in coder_dict[coder].keys():
                        coder_dict[coder][(tok_id, tok_id + 1)].append(cat)
                    else:
                        coder_dict[coder][(tok_id, tok_id + 1)] = [cat]

        # Span Based Detection
        elif evaluation_type == "ds":
            if (start, end) in coder_dict[coder].keys():
                coder_dict[coder][(start, end)].append(cat)
            else:
                coder_dict[coder][(start, end)] = [cat]

        # Span Based Correction
        else:
            # With error type classification
            if evaluation_type == "cse":
                if (start, end, cat, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cat, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cat, cor)] = [cat]
            # Without error type classification
            else:
                if (start, end, cor) in coder_dict[coder].keys():
                    coder_dict[coder][(start, end, cor)].append(cat)
                else:
                    coder_dict[coder][(start, end, cor)] = [cat]
    return coder_dict


def evaluate_edits(hyp_dict, ref_dict, best, sent_id, original_sentence, beta, verbose):
    # Verbose output: display the original sentence
    if verbose:
        print("{:-^40}".format(""))
        print("Original sentence " + str(sent_id) + ": " + original_sentence)
    # Store the best sentence level scores and hyp+ref combination IDs
    # best_f is initialised as -1 cause 0 is a valid result.
    best_tp, best_fp, best_fn, best_f, best_hyp, best_ref = 0, 0, 0, -1, 0, 0
    best_cat = {}
    # Compare each hyp and ref combination
    for hyp_id in hyp_dict.keys():
        for ref_id in ref_dict.keys():
            # Get the local counts for the current combination.
            tp, fp, fn, cat_dict = compareEdits(hyp_dict[hyp_id], ref_dict[ref_id])
            # Compute the local sentence scores (for verbose output only)
            loc_p, loc_r, loc_f = computeFScore(tp, fp, fn, beta)
            # Compute the global sentence scores
            p, r, f = computeFScore(
                tp + best["tp"], fp + best["fp"], fn + best["fn"], beta
            )
            # Save the scores if they are better in terms of:
            # 1. Higher F-score
            # 2. Same F-score, higher TP
            # 3. Same F-score and TP, lower FP
            # 4. Same F-score, TP and FP, lower FN
            if (
                (f > best_f)
                or (f == best_f and tp > best_tp)
                or (f == best_f and tp == best_tp and fp < best_fp)
                or (f == best_f and tp == best_tp and fp == best_fp and fn < best_fn)
            ):
                best_tp, best_fp, best_fn = tp, fp, fn
                best_f, best_hyp, best_ref = f, hyp_id, ref_id
                best_cat = cat_dict
            # Verbose output
            if verbose:
                # Prepare verbose output edits.
                hyp_verb = list(sorted(hyp_dict[hyp_id].keys()))
                ref_verb = list(sorted(ref_dict[ref_id].keys()))
                # add categories hyp_dict[hyp_id] looks like (0, 1, "str") hyp_dict[
                # hyp_id][h] is a list, always length one, of the corresponding category
                hyp_verb = [h + (hyp_dict[hyp_id][h][0],) for h in hyp_verb]
                ref_verb = [r + (ref_dict[ref_id][r][0],) for r in ref_verb]
                # Ignore noop edits
                if not hyp_verb or hyp_verb[0][0] == -1:
                    hyp_verb = []
                if not ref_verb or ref_verb[0][0] == -1:
                    ref_verb = []
                # Print verbose info
                print("{:-^40}".format(""))
                print(
                    "SENTENCE "
                    + str(sent_id)
                    + " - HYP "
                    + str(hyp_id)
                    + " - REF "
                    + str(ref_id)
                )
                print("HYPOTHESIS EDITS :", hyp_verb)
                print("REFERENCE EDITS  :", ref_verb)
                print("Local TP/FP/FN   :", str(tp), str(fp), str(fn))
                print(
                    "Local P/R/F" + str(beta) + "  :",
                    str(loc_p),
                    str(loc_r),
                    str(loc_f),
                )
                print(
                    "Global TP/FP/FN  :",
                    str(tp + best["tp"]),
                    str(fp + best["fp"]),
                    str(fn + best["fn"]),
                )
                print("Global P/R/F" + str(beta) + "  :", str(p), str(r), str(f))

    # Verbose output: display the best hyp+ref combination
    if verbose:
        print("{:-^40}".format(""))
        print(
            "^^ HYP "
            + str(best_hyp)
            + ", REF "
            + str(best_ref)
            + " chosen for sentence "
            + str(sent_id)
        )
        print("Local results:")
        header = ["Category", "TP", "FP", "FN"]
        body = [[k, *v] for k, v in best_cat.items()]
        print_table([header] + body)

    # Save the best TP, FP and FNs as a dict, and return this and the best_cat dict
    best_dict = {"tp": best_tp, "fp": best_fp, "fn": best_fn}
    return best_dict, best_cat


def score(
    hyp_m2,
    ref_m2,
    beta=0.5,
    verbose=False,
    evaluation_type="cs",
    single=False,
    multi=False,
    filt=None,
):
    # Store global corpus level best counts here
    best_dict = Counter({"tp": 0, "fp": 0, "fn": 0})
    best_cats = {}

    # Process each sentence
    sents = zip(hyp_m2, ref_m2)
    for sent_id, sent in enumerate(sents):
        # Simplify the edits into lists of lists
        hyp_edits = simplify_edits(sent[0])
        ref_edits = simplify_edits(sent[1])

        # Process the edits for detection/correction based on args
        hyp_dict = process_edits(hyp_edits, evaluation_type, single, multi, filt)
        ref_dict = process_edits(ref_edits, evaluation_type, single, multi, filt)

        # original sentence for logging
        original_sentence = sent[0][2:].split("\nA")[0]

        # Evaluate edits and get best TP, FP, FN hyp+ref combo.
        count_dict, cat_dict = evaluate_edits(
            hyp_dict, ref_dict, best_dict, sent_id, original_sentence, beta, verbose
        )

        # Merge these dicts with best_dict and best_cats
        best_dict += Counter(count_dict)
        best_cats = merge_dict(best_cats, cat_dict)
    return best_dict, best_cats


def to_m2(lines, tok=False, lev=False, merge="rules"):
    # Load Errant
    annotator = errant.load("en")

    out = []
    for line in lines:
        orig, cors = line[0], line[1:]

        # Parse orig with spacy
        orig = annotator.parse(orig, tok)

        # Write orig to the output m2 file
        out.append(" ".join(["S"] + [token.text for token in orig]) + "\n")

        for cor_id, cor in enumerate(cors):
            # If the texts are the same, write a noop edit
            if orig.text.strip() == cor:
                out[-1] += noop_edit(cor_id) + "\n"

            # Otherwise, do extra processing
            else:
                # Parse cor with spacy
                cor = annotator.parse(cor, tok)

                # Align the texts and extract and classify the edits
                edits = annotator.annotate(orig, cor, lev, merge)
                for edit in edits:
                    # Write the edit to the output m2 file
                    out[-1] += edit.to_m2(cor_id) + "\n"

        # Remove trailing newlines when we have processed all corrections for each line
        out[-1] = out[-1].rstrip("\n")
    return out


def print_results(best, best_cats, evaluation_type="cs", cat=None, beta=0.5):
    # Prepare output title.
    if evaluation_type == "dt":
        title = " Token-Based Detection "
    elif evaluation_type == "ds":
        title = " Span-Based Detection "
    elif evaluation_type == "cse":
        title = " Span-Based Correction + Classification "
    else:
        title = " Span-Based Correction "

    # Category Scores
    if cat:
        best_cats = processCategories(best_cats, cat)
        print("")
        print("{:=^66}".format(title))
        print(
            "Category".ljust(14),
            "TP".ljust(8),
            "FP".ljust(8),
            "FN".ljust(8),
            "P".ljust(8),
            "R".ljust(8),
            "F" + str(beta),
        )
        for cat, cnts in sorted(best_cats.items()):
            cat_p, cat_r, cat_f = computeFScore(cnts[0], cnts[1], cnts[2], beta)
            print(
                cat.ljust(14),
                str(cnts[0]).ljust(8),
                str(cnts[1]).ljust(8),
                str(cnts[2]).ljust(8),
                str(cat_p).ljust(8),
                str(cat_r).ljust(8),
                cat_f,
            )

    # Print the overall results.
    print("")
    print("{:=^46}".format(title))
    print("\t".join(["TP", "FP", "FN", "Prec", "Rec", "F" + str(beta)]))
    print(
        "\t".join(
            map(
                str,
                [best["tp"], best["fp"], best["fn"]]
                + list(computeFScore(best["tp"], best["fp"], best["fn"], beta)),
            )
        )
    )
    print("{:=^46}".format(""))
    print("")
