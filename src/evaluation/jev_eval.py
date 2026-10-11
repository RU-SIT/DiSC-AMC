"""Jev (TypeSafe) evaluation provider for modulation classification, via BeatAPI.

Jev is a "choice" classifier, not a free-text generator: you give it a state
(our prompt) plus a labeled option set, and it returns the highest-probability
option with a probability per option. That is exactly DiSC-AMC's native
interface — each prompt is already narrowed to a k_top-class shortlist
(``data['k-top']`` / ``prompt_data['candidates']``, see
:func:`src.evaluation.utils.build_prompts_data`) before it ever reaches an
LLM. So unlike the Gemini/OpenAI providers, there's no chain-of-thought to
parse: the returned option *is* the label.

Access is through BeatAPI (https://docs.beatapi.io/decisions), a proxy with
request/response schemas identical to TypeSafe's own API — only the base URL
and model name differ. Default model here is the paid ``jev-1.13``
($0.042/M input tokens, $0 output) since it runs at the account's normal
rate limit instead of the free ``jev-1.13-free`` tier's 1-10 req/min cap;
pass ``model_name="jev-1.13-free"`` to fall back to the zero-cost tier.

Requests are dispatched concurrently (``max_workers`` threads, default 10) via
a thread pool rather than one-at-a-time: each call is a single lightweight
synchronous HTTP round trip (no streaming/long generation to wait on), so
fanning them out is what actually keeps a multi-thousand-query sweep to
minutes rather than hours. A 429 on any one request backs off and retries
without blocking the other in-flight requests.
"""

import os
import re
import sys
import time
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import requests

from src.naming import ExperimentConfig, eval_result_name

from .utils import (
    load_data, build_prompts_data, load_existing_results,
    get_prompts_to_process, save_results_atomic, build_result_entry,
    sort_results_by_prompt, get_unique_prompts, print_metrics,
)

load_dotenv()

JEV_API_URL = "https://api.beatapi.io/v1/systemone"


def _get_api_key() -> str:
    return os.getenv("BEATAPI_API_KEY") or os.getenv("beatapi") or ""

CLASS_NAMES = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']

RADIOML_CLASS_NAMES = [
    '128APSK', '128QAM', '16APSK', '16PSK', '16QAM', '256QAM',
    '32APSK', '32PSK', '32QAM', '4ASK', '64APSK', '64QAM',
    '8ASK', '8PSK', 'AM-DSB-SC', 'AM-DSB-WC', 'AM-SSB-SC', 'AM-SSB-WC',
    'BPSK', 'FM', 'GMSK', 'OOK', 'OQPSK', 'QPSK',
]

def get_class_names(dataset_type: str = 'own') -> list:
    """Return class names for the given dataset type."""
    if dataset_type == 'radioml':
        return RADIOML_CLASS_NAMES
    return CLASS_NAMES


QUESTION_ID = "modulation"

# Per-class descriptions for Jev's "choice" criteria. BeatAPI's /decisions
# schema defines ``criteria`` as "an object mapping each option name to what
# it means" (a description string) — not the bare class-name-only mapping
# this module originally sent (``{c: None for c in candidates}``). Without a
# description, Jev's choice is grounded only in whatever generic modulation
# knowledge happens to be in `state`; these blurbs give the decision itself
# the discriminating signal (envelope/phase/amplitude behavior) its schema
# is designed to use. Covers both CLASS_NAMES (SynthMod-10) and
# RADIOML_CLASS_NAMES; overlapping keys (4ASK, 8ASK, GMSK, OOK, OQPSK) share
# one entry since they're the same modulation scheme in both datasets.
CLASS_DESCRIPTIONS = {
    # --- SynthMod-10 ---
    '4ASK': "4-level Amplitude Shift Keying: constant phase, amplitude-only "
            "modulation with 4 discrete amplitude levels, no phase or "
            "frequency variation.",
    '4PAM': "4-level Pulse Amplitude Modulation: 4 equally spaced bipolar "
            "amplitude levels, no phase or frequency variation; similar to "
            "4ASK but symmetric around zero.",
    '8ASK': "8-level Amplitude Shift Keying: constant phase, amplitude-only "
            "modulation with 8 discrete amplitude levels (higher-order than "
            "4ASK), no phase or frequency variation.",
    '16PAM': "16-level Pulse Amplitude Modulation: 16 equally spaced bipolar "
             "amplitude levels, a much finer-grained amplitude histogram "
             "than 4PAM/8ASK, no phase or frequency variation.",
    'CPFSK': "Continuous Phase Frequency Shift Keying: constant envelope "
             "(no amplitude variation), phase changes smoothly and "
             "continuously between frequency tones, without the Gaussian "
             "pulse shaping used by GFSK/GMSK.",
    'DQPSK': "Differential Quadrature Phase Shift Keying: constant envelope, "
             "4-point phase constellation where the relative (differential) "
             "phase shift between symbols carries the data, not the "
             "absolute phase.",
    'GFSK': "Gaussian Frequency Shift Keying: constant envelope, binary "
            "frequency shift with Gaussian-filtered pulse shaping that "
            "smooths spectral sidelobes relative to raw FSK/CPFSK.",
    'GMSK': "Gaussian Minimum Shift Keying: constant envelope, continuous-"
            "phase binary FSK with Gaussian pre-filtering and minimum "
            "frequency spacing (modulation index 0.5); used in GSM; the "
            "narrowest spectral occupancy of the FSK family here.",
    'OQPSK': "Offset Quadrature Phase Shift Keying: constant envelope, "
             "4-phase modulation like QPSK but with the I and Q streams "
             "offset by half a symbol period, avoiding abrupt 180-degree "
             "phase transitions.",
    'OOK': "On-Off Keying: the simplest amplitude modulation — binary "
           "(on/off) amplitude with no phase or frequency variation; "
           "effectively 2-level ASK with one level at zero.",
    # --- RadioML.2018.01a additions ---
    '128APSK': "Amplitude-Phase Shift Keying with 128 symbols spread across "
               "multiple concentric amplitude rings, each with several "
               "phase points; very dense combined amplitude+phase "
               "constellation.",
    '128QAM': "Quadrature Amplitude Modulation with 128 symbols on a dense "
              "square/cross grid varying both amplitude and phase; high "
              "spectral efficiency, closely packed constellation points.",
    '16APSK': "Amplitude-Phase Shift Keying with 16 symbols, typically on "
              "two concentric amplitude rings with phase variation; common "
              "in satellite links (e.g. DVB-S2).",
    '16PSK': "Phase Shift Keying with 16 equally spaced phase states at "
             "constant amplitude (constant envelope), no amplitude "
             "variation.",
    '16QAM': "Quadrature Amplitude Modulation with 16 symbols on a 4x4 "
             "square grid, varying both amplitude and phase.",
    '256QAM': "Quadrature Amplitude Modulation with 256 symbols on a dense "
              "16x16 grid; the finest amplitude/phase resolution and "
              "highest-order QAM in this set.",
    '32APSK': "Amplitude-Phase Shift Keying with 32 symbols across several "
              "concentric amplitude rings with phase variation.",
    '32PSK': "Phase Shift Keying with 32 equally spaced phase states at "
             "constant amplitude, no amplitude variation.",
    '32QAM': "Quadrature Amplitude Modulation with 32 symbols combining "
             "amplitude and phase variation, denser than 16QAM.",
    '64APSK': "Amplitude-Phase Shift Keying with 64 symbols across several "
              "concentric amplitude rings with phase variation.",
    '64QAM': "Quadrature Amplitude Modulation with 64 symbols on an 8x8 "
             "grid, denser than 16QAM.",
    '8PSK': "Phase Shift Keying with 8 equally spaced phase states at "
            "constant amplitude, no amplitude variation.",
    'AM-DSB-SC': "Analog Double-Sideband Suppressed-Carrier AM: no discrete "
                 "carrier tone, symmetric sidebands, continuous (not "
                 "symbol-discrete) envelope variation.",
    'AM-DSB-WC': "Analog Double-Sideband AM With Carrier (standard AM): a "
                 "strong discrete carrier tone plus symmetric sidebands.",
    'AM-SSB-SC': "Analog Single-Sideband Suppressed-Carrier AM: only one "
                 "sideband transmitted, no carrier tone — the most "
                 "bandwidth-efficient analog AM variant here.",
    'AM-SSB-WC': "Analog Single-Sideband AM With Carrier: one sideband plus "
                 "a discrete carrier tone.",
    'BPSK': "Binary Phase Shift Keying: the simplest phase modulation, 2 "
            "phase states (0 / 180 degrees) at constant amplitude, no "
            "amplitude variation.",
    'FM': "Analog Frequency Modulation: continuous frequency variation "
          "proportional to the message signal, constant envelope, no "
          "discrete symbol or phase states.",
    'QPSK': "Quadrature Phase Shift Keying: constant-envelope 4-phase "
            "modulation (0/90/180/270 degrees), no amplitude variation and "
            "no I/Q timing offset (unlike OQPSK).",
}


def tailor_prompt_for_jev(prompt: str) -> str:
    """Strip the free-text-generation instructions that are dead weight for Jev.

    The shared prompt builder (``src/prompt/data_processing.py``) targets
    generative LLMs: it mandates ``<think>`` CoT tags and an output-format
    contract ("single entry, no extra words", ...), then repeats "following
    all rules above" in the final query. Jev never free-text-generates or
    gets parsed for those tags — it returns a structured choice + probability
    distribution via a separate ``questions``/``criteria`` schema (see
    :func:`get_jev_response`). Those instructions are pure inert context for
    Jev, so this removes the "**RESPONSE RULES:**" block and the trailing
    "Classify the signal... following all rules above" sentence, leaving the
    role/context/exemplars/query intact.
    """
    prompt = re.sub(
        r"\n*\*\*RESPONSE RULES:\*\*.*?\n-----\n",
        "\n",
        prompt,
        flags=re.DOTALL,
    )
    prompt = re.sub(
        r"\n\*\*Answer:\*\*\s*\nClassify the signal based on.*\Z",
        "",
        prompt,
        flags=re.DOTALL,
    )
    return prompt.rstrip() + "\n"


def get_jev_response(session: requests.Session, prompt: str, candidates: list,
                     model: str = "jev-1.13",
                     criteria_descriptions: dict | None = None) -> dict:
    """Get a choice response from the Jev API (via BeatAPI) with retry logic.

    Returns the raw ``answers[QUESTION_ID]`` dict, e.g.::

        {"type": "choice", "choice": "OOK",
         "probabilities": {"OOK": 0.88, "4ASK": 0.12, ...}, "confidence": 0.81}

    A 429 (rate limit, expected often on ``jev-1.13-free``) is retried after
    honoring the ``Retry-After`` header rather than counting against the
    normal 3-attempt budget, since it's an expected/transient condition, not
    an error.

    Parameters
    ----------
    criteria_descriptions : dict, optional
        Maps candidate class name -> a short description of what it means
        (BeatAPI's documented ``criteria`` schema: "an object mapping each
        option name to what it means", see https://docs.beatapi.io/decisions).
        Defaults to ``None`` per candidate when not given, matching this
        module's original (undocumented/degenerate) behavior.
    """
    criteria_descriptions = criteria_descriptions or {}
    payload = {
        "state": prompt,
        "model": model,
        "questions": {
            QUESTION_ID: {
                "type": "choice",
                "instructions": "Select the modulation class that best matches the described signal.",
                "criteria": {c: criteria_descriptions.get(c) for c in candidates},
            }
        },
    }
    attempt = 0
    while attempt < 3:
        try:
            response = session.post(JEV_API_URL, json=payload, timeout=30)
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After", 60))
                print(f"Rate limited, waiting {wait:.0f}s...")
                time.sleep(wait)
                continue  # doesn't count against the attempt budget
            response.raise_for_status()
            return response.json()["answers"][QUESTION_ID]
        except Exception as e:
            attempt += 1
            print(f"Attempt {attempt} failed: {e}")
            if attempt == 3:
                raise
    return {}


def _output_path(cfg: ExperimentConfig, prompt_type: str, model_name: str) -> str:
    return eval_result_name(cfg, prompt_type, model_name, "jev")


def _run_job(session: requests.Session, job: tuple, api_model_name: str,
             criteria_descriptions: dict | None = None) -> dict:
    filename, try_idx, prompt, candidates = job
    answer = get_jev_response(session, prompt, candidates, model=api_model_name,
                              criteria_descriptions=criteria_descriptions)
    entry = build_result_entry(filename, try_idx, prompt, answer.get('choice', ''))
    entry['probabilities'] = answer.get('probabilities', {})
    entry['confidence'] = answer.get('confidence')
    return entry


def main(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
         model_name="jev-1.13", noise_mode='noisySignal',
         n_bins=10, top_k=5, num_tries=3, prediction_source='dnn',
         feature_type='stats', n_components=0,
         ood_train_folder='', use_rag=False, rag_k=0,
         output_dir='.', prompt_version='v1', backbone='dino',
         dataset_type='own', max_workers=10, save_every=25,
         data_root=None, tailor_prompt=False, api_model_name=None,
         use_criteria_descriptions=False):
    """
    Parameters
    ----------
    model_name : str
        Used to tag the output results filename (via :func:`_output_path`).
        Pass e.g. ``"jev-1.13-tailored"`` for a tailored-prompt run so it
        doesn't clobber the default-prompt run's results file.
    tailor_prompt : bool
        When True, strips the generative-LLM CoT/format instructions from
        each prompt via :func:`tailor_prompt_for_jev` before sending it to
        Jev, since they're inert for its structured-choice API.
    api_model_name : str, optional
        The actual model string sent to the Jev/BeatAPI request. Defaults
        to ``model_name`` when not given (so plain runs are unaffected);
        set this explicitly when ``model_name`` has been changed purely to
        tag the output file (e.g. the "-tailored" suffix).
    use_criteria_descriptions : bool
        When True, populates each candidate's ``criteria`` entry in the Jev
        request with a short description of that modulation scheme (see
        :data:`CLASS_DESCRIPTIONS`) instead of ``None``, matching BeatAPI's
        documented schema ("an object mapping each option name to what it
        means"). Independent of ``tailor_prompt`` — this changes the
        request payload, not the ``state`` prompt text.
    """
    api_model_name = api_model_name or model_name
    criteria_descriptions = CLASS_DESCRIPTIONS if use_criteria_descriptions else None
    cfg = ExperimentConfig(
        dataset_folder=dataset_folder,
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_type=feature_type,
        n_components=n_components,
        ood_train_folder=ood_train_folder,
        use_rag=use_rag,
        rag_k=rag_k,
        prompt_version=prompt_version,
        backbone=backbone,
    )
    results = []
    results_lock = threading.Lock()
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))
    fallback_candidates = get_class_names(dataset_type)

    try:
        root = data_root or '../../data/own'
        data, _, _ = load_data(f'{root}/{dataset_folder}', noise_mode, n_bins, top_k,
                               prediction_source=prediction_source,
                               feature_tag=f'emb{n_components}' if feature_type == 'embeddings' else '',
                               backbone=backbone)

        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json",
        })
        adapter = requests.adapters.HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers)
        session.mount("https://", adapter)

        all_prompts_data = build_prompts_data(data, prompt_type)
        if tailor_prompt:
            for prompt_data in all_prompts_data:
                prompt_data['prompt'] = tailor_prompt_for_jev(prompt_data['prompt'])
        results, prompts_done, completed = load_existing_results(filepath, num_tries, key_field='prompt')
        prompts_to_process = get_prompts_to_process(all_prompts_data, completed, prompts_done, key_field='prompt')

        jobs = [
            (prompt_data['filename'], i, prompt_data['prompt'], prompt_data.get('candidates') or fallback_candidates)
            for prompt_data, num_done in prompts_to_process
            for i in range(num_done, num_tries)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_job, session, job, api_model_name, criteria_descriptions): job
                for job in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures)):
                filename, try_idx, _, _ = futures[future]
                try:
                    entry = future.result()
                    with results_lock:
                        results.append(entry)
                        if len(results) % save_every == 0:
                            save_results_atomic(results, filepath)
                except Exception as e:
                    print(f"Error processing {filename} (try {try_idx}): {e}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial results...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        save_results_atomic(results, filepath)


def read_results(dataset_folder='unlabeled_10k', prompt_type='discret_prompts',
                 model_name='jev-1.13', noise_mode='noisySignal',
                 n_bins=5, top_k=5, prediction_source='dnn',
                 feature_type='stats', n_components=0,
                 ood_train_folder='', use_rag=False, rag_k=0,
                 output_dir='.', prompt_version='v1', backbone='dino'):
    """Read results from JSON file."""
    import json
    cfg = ExperimentConfig(
        dataset_folder=dataset_folder,
        prediction_source=prediction_source,
        noise_mode=noise_mode,
        n_bins=n_bins,
        top_k=top_k,
        feature_type=feature_type,
        n_components=n_components,
        ood_train_folder=ood_train_folder,
        use_rag=use_rag,
        rag_k=rag_k,
        prompt_version=prompt_version,
        backbone=backbone,
    )
    filepath = os.path.join(output_dir, _output_path(cfg, prompt_type, model_name))
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    DATASET_FOLDER = 'unlabeled_10k'
    PROMPT_TYPE = 'discret_prompts'
    MODEL_NAME = "jev-1.13"
    NOISE_MODE = 'noisySignal'
    N_BINS, TOP_K, NUM_TRIES = 10, 5, 3

    main(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K, NUM_TRIES)
    results = read_results(DATASET_FOLDER, PROMPT_TYPE, MODEL_NAME, NOISE_MODE, N_BINS, TOP_K)
    sorted_results = sort_results_by_prompt(results)
    print(f"Unique prompts: {len(get_unique_prompts(results))}")
    print_metrics(sorted_results, CLASS_NAMES)
