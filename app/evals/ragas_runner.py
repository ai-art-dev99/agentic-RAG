import json
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate

def load_samples(path="evals/samples.jsonl"):
    return [json.loads(l) for l in open(path)]

if __name__ == "__main__":
    dataset = load_samples()
    # each item: {"question":..., "contexts":[...], "answer":...}
    result = evaluate(dataset=dataset,
                      metrics=[faithfulness, answer_relevancy, context_precision])
    print(result)
