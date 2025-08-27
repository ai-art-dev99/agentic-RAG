from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_small_batch():
    metric_f = FaithfulnessMetric()
    metric_a = AnswerRelevancyMetric()
    metric_c = ContextualRelevancyMetric()

    cases = [
      LLMTestCase(input="What is OpenSearch hybrid search?",
                  actual_output="Hybrid search combines BM25 and vector search.",
                  retrieval_context=["Hybrid search combines keyword and semantic search."]),
    ]
    for case in cases:
        assert metric_f.measure(case) >= 0.5
        assert metric_a.measure(case) >= 0.5
        assert metric_c.measure(case) >= 0.5
