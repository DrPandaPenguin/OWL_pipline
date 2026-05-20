import copy


DEFAULT_CONFIG = {
 # --- Pipeline method ---
    "pipeline": "slide_anchored",          # "slide_anchored" | "multi_stage" | "single_pass" | "direct"

 # --- Phase 1: KU extraction ---
    "ku_model": "gpt-5.2",
    "ku_temperature": 0.1,
    "ku_system_prompt": None,           # None = use PHASE1_SYSTEM from extract_nodes.py
    "ku_user_template": None,           # None = use PHASE1_USER_TEMPLATE
    "ku_method": "atomic",             # "atomic" | "discourse" | "sentence"
    "ku_match_threshold": 0.5,         # fuzzy match for KU-to-sentence mapping

 # --- Phase 3: Node construction ---
    "node_model": "gpt-5.2",
    "node_temperature": 0.2,
    "node_system_prompt": None,        # None = use PHASE3_SYSTEM
    "node_user_template": None,        # None = use PHASE3_USER_TEMPLATE

 # --- Phase 4: Strict edges ---
    "strict_model": "gpt-5.2",
    "strict_temperature": 0.1,
    "strict_system_prompt": None,      # None = use default
    "strict_similarity_threshold": 0.8,
    "include_strict": True,

 # --- Phase 5: Soft edges ---
    "soft_model": "gpt-5.2",
    "soft_temperature": 0.2,
    "soft_system_prompt": None,
    "soft_default_confidence": 0.7,
    "include_soft": True,

 # --- Edge Refinement pass (multi_stage_refined pipeline) ---
    "refine_temperature": 0.1,         # temperature for the refinement LLM call

 # --- Slide structure (slide_structure / slide_anchored* pipelines) ---
    "slide_text": None,                # raw slide text; if None falls back to multi_stage
    "skip_slide_parse": False,         # True = use Python markdown parser (no LLM call);
 # False = use LLM to parse slide structure (default)

 # --- Graph enrichment pass (optional post-extraction step) ---
    "enrich_graph": False,             # True = run enrich_graph() after pipeline completes;
 # adds description + why_it_matters to nodes,
 # adds reason to edges. Off by default (extra LLM call).

 # --- Edge types ---
    "edge_types": [
        "defines", "requires", "explains", "details",
        "example_of", "contrasts", "drives",
    ],

 # --- Experiment control ---
    "num_runs": 3,                     # repeat count for variance
    "transcript_paths": [
        "data/Test_transcript_lec4_Ecom",
    ],
    "slide_text_path": "data/Test_Slide_strcuture_lec4_Ecoms",
}


# EXPERIMENTS: 각 항목이 DEFAULT_CONFIG 위에 덮어씀
EXPERIMENTS = {

 # Exp 1: single-pass vs multi-stage vs refined
    "exp1_multi_stage": {
        "pipeline": "multi_stage",
        "group": "exp1_single_vs_multi",
    },
    "exp1_single_pass": {
        "pipeline": "single_pass",
        "group": "exp1_single_vs_multi",
    },

 # Exp 2: KU Method
    "exp2_atomic": {
        "ku_method": "atomic",
        "group": "exp2_ku_method",
    },
    "exp2_sentence": {
        "ku_method": "sentence",
        "group": "exp2_ku_method",
    },

 # Exp 3: Strict/Soft combination
    "exp3_strict_only": {
        "include_strict": True,
        "include_soft": False,
        "group": "exp3_strict_soft",
    },
    "exp3_soft_only": {
        "include_strict": False,
        "include_soft": True,
        "group": "exp3_strict_soft",
    },
    "exp3_both": {
        "include_strict": True,
        "include_soft": True,
        "group": "exp3_strict_soft",
    },

 # Exp 4: Edge Types
    "exp4_7types": {
        "edge_types": [
            "defines", "requires", "explains", "details",
            "example_of", "contrasts", "drives",
        ],
        "group": "exp4_edge_types",
    },
    "exp4_8types": {
        "edge_types": [
            "defines", "requires", "explains", "details",
            "example_of", "contrasts", "drives", "precedes",
        ],
        "group": "exp4_edge_types",
    },
    "exp4_4types": {
        "edge_types": [
            "defines", "requires", "explains", "example_of",
        ],
        "group": "exp4_edge_types",
    },

 # Exp 5: Slide structure (requires slide_text override at runtime)
 # Run with: python -m experiments.run_experiment exp5_slide_structure
 # (slide_text must be injected via config override or set here)
    "exp5_multi_stage": {
        "pipeline": "multi_stage",
        "group": "exp5_pipeline_variants",
    },

 # Exp 6: Slide grounding vs KU grounding comparison
 # Research question: does slide-segment grounding outperform KU-sentence grounding?
 # All four conditions require slide_text to be set at runtime for slide_* pipelines.
 #
 # Run: python -m experiments.run_experiment --group exp6_slide_vs_ku --runs 3
    "exp6_multi_stage": {
        "pipeline": "multi_stage",
        "group": "exp6_slide_vs_ku",
    },
    "exp6_slide_anchored": {
        "pipeline": "slide_anchored",
        "slide_text": None,          # inject at runtime
        "group": "exp6_slide_vs_ku",
    },

 # General: Temperature variations
    "temp_low": {
        "ku_temperature": 0.0,
        "node_temperature": 0.0,
        "strict_temperature": 0.0,
        "soft_temperature": 0.0,
        "group": "temperature",
    },
    "temp_default": {
        "group": "temperature",
    },
    "temp_high": {
        "ku_temperature": 0.5,
        "node_temperature": 0.5,
        "strict_temperature": 0.3,
        "soft_temperature": 0.5,
        "group": "temperature",
    },

 # Exp 7: Segmentation method comparison
 # Research question: which transcript segmentation method produces the best mini-graphs?
 # All use merge strategy "c" (Python dedup + cross-PART LLM) to isolate segmentation effect.
 # Baseline: slide_anchored (no segmentation, full transcript)
    "exp7_baseline": {
        "pipeline": "slide_anchored",
        "slide_text": None,
        "group": "exp7_segmentation",
    },

 # Exp 9: Model comparison
 # Research question: does the choice of GPT model affect graph quality?
 # Fixed pipeline: slide_anchored (current best). 1 run each to compare.
    "exp9_gpt5.2": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-5.2",
        "node_model": "gpt-5.2",
        "strict_model": "gpt-5.2",
        "soft_model": "gpt-5.2",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },
    "exp9_gpt5": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-5",
        "node_model": "gpt-5",
        "strict_model": "gpt-5",
        "soft_model": "gpt-5",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },
    "exp9_gpt5.1": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-5.1",
        "node_model": "gpt-5.1",
        "strict_model": "gpt-5.1",
        "soft_model": "gpt-5.1",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },
    "exp9_gpt4o": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-4o",
        "node_model": "gpt-4o",
        "strict_model": "gpt-4o",
        "soft_model": "gpt-4o",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },
    "exp9_gpt4o_mini": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-4o-mini",
        "node_model": "gpt-4o-mini",
        "strict_model": "gpt-4o-mini",
        "soft_model": "gpt-4o-mini",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },
    "exp9_gpt4_turbo": {
        "pipeline": "slide_anchored",
        "ku_model": "gpt-4-turbo",
        "node_model": "gpt-4-turbo",
        "strict_model": "gpt-4-turbo",
        "soft_model": "gpt-4-turbo",
        "slide_text": None,
        "group": "exp9_model_comparison",
    },

 # Exp 9b: Model comparison (transcript-only, multi_stage pipeline)
 # Same models as exp9, but WITHOUT slides tests model capability on unstructured input.
    "exp9b_gpt5.2_transcript": {
        "pipeline": "multi_stage",
        "ku_model": "gpt-5.2",
        "node_model": "gpt-5.2",
        "strict_model": "gpt-5.2",
        "soft_model": "gpt-5.2",
        "group": "exp9b_model_transcript",
    },
    "exp9b_gpt5.1_transcript": {
        "pipeline": "multi_stage",
        "ku_model": "gpt-5.1",
        "node_model": "gpt-5.1",
        "strict_model": "gpt-5.1",
        "soft_model": "gpt-5.1",
        "group": "exp9b_model_transcript",
    },
    "exp9b_gpt4o_transcript": {
        "pipeline": "multi_stage",
        "ku_model": "gpt-4o",
        "node_model": "gpt-4o",
        "strict_model": "gpt-4o",
        "soft_model": "gpt-4o",
        "group": "exp9b_model_transcript",
    },
    "exp9b_gpt4o_mini_transcript": {
        "pipeline": "multi_stage",
        "ku_model": "gpt-4o-mini",
        "node_model": "gpt-4o-mini",
        "strict_model": "gpt-4o-mini",
        "soft_model": "gpt-4o-mini",
        "group": "exp9b_model_transcript",
    },
    "exp9b_gpt4_turbo_transcript": {
        "pipeline": "multi_stage",
        "ku_model": "gpt-4-turbo",
        "node_model": "gpt-4-turbo",
        "strict_model": "gpt-4-turbo",
        "soft_model": "gpt-4-turbo",
        "group": "exp9b_model_transcript",
    },

 # Exp 9c: Model comparison (DirectExtract no KU, no slides)
    "exp9c_gpt5.2_direct": {
        "pipeline": "direct",
        "ku_model": "gpt-5.2", "node_model": "gpt-5.2",
        "strict_model": "gpt-5.2", "soft_model": "gpt-5.2",
        "group": "exp9c_model_direct",
    },
    "exp9c_gpt5.1_direct": {
        "pipeline": "direct",
        "ku_model": "gpt-5.1", "node_model": "gpt-5.1",
        "strict_model": "gpt-5.1", "soft_model": "gpt-5.1",
        "group": "exp9c_model_direct",
    },
    "exp9c_gpt4o_direct": {
        "pipeline": "direct",
        "ku_model": "gpt-4o", "node_model": "gpt-4o",
        "strict_model": "gpt-4o", "soft_model": "gpt-4o",
        "group": "exp9c_model_direct",
    },
    "exp9c_gpt4o_mini_direct": {
        "pipeline": "direct",
        "ku_model": "gpt-4o-mini", "node_model": "gpt-4o-mini",
        "strict_model": "gpt-4o-mini", "soft_model": "gpt-4o-mini",
        "group": "exp9c_model_direct",
    },
    "exp9c_gpt4_turbo_direct": {
        "pipeline": "direct",
        "ku_model": "gpt-4-turbo", "node_model": "gpt-4-turbo",
        "strict_model": "gpt-4-turbo", "soft_model": "gpt-4-turbo",
        "group": "exp9c_model_direct",
    },

 # Exp 8: Merge strategy comparison
 # Research question: which merge strategy produces better cross-PART edges?
 # Fixed segmentation: TF-IDF (cheapest, no LLM call for segmentation).
}


def merge_config(experiment_name):
    base = copy.deepcopy(DEFAULT_CONFIG)
    if experiment_name == "default":
        return base
    overrides = EXPERIMENTS.get(experiment_name)
    if overrides is None:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(EXPERIMENTS.keys())}")
    base.update(overrides)
    return base


def list_experiments(group=None):
    if group is None:
        return list(EXPERIMENTS.keys())
    return [name for name, cfg in EXPERIMENTS.items() if cfg.get("group") == group]


def list_groups():
    return sorted({cfg.get("group", "ungrouped") for cfg in EXPERIMENTS.values()})
