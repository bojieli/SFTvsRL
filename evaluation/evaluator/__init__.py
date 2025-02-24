from evaluation.evaluator.llama_evaluator import LlamaEvaluator
from evaluation.evaluator.api_evaluator import APIEvaluator
evaluator_init = {
    # "ApiEvaluator": ApiEvaluator,
    # "OpenAIEvaluator": OpenAIEvaluator,
    # "OpenAIEvaluator_oneline": OpenAIEvaluator_oneline,
    # # "CambrianEvaluator": CambrianEvaluator,
    # "CambrianEvaluator_oneline": CambrianEvaluator_oneline,
    # "LlamaEvaluator_oneline": LlamaEvaluator_oneline,
    'APIEvaluator': APIEvaluator,
    "LlamaEvaluator": LlamaEvaluator
}