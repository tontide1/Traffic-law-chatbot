from backend.core.query_router import QueryClass, classify_query, retrieval_policy_for


def test_classifies_direct_definition_question():
    question = "Hành lang an toàn đường bộ được xác định từ đâu và nhằm mục đích gì?"

    assert classify_query(question) == QueryClass.DIRECT_DEFINITION


def test_classifies_relational_question():
    question = "Quy định về hành lang an toàn đường bộ liên quan đến trách nhiệm của cơ quan nào?"

    assert classify_query(question) == QueryClass.RELATIONAL


def test_classifies_direct_rule_question():
    question = "Người điều khiển xe máy vi phạm nồng độ cồn bị phạt bao nhiêu?"

    assert classify_query(question) == QueryClass.DIRECT_RULE


def test_defaults_uncertain_question_to_open_ended():
    question = "Hãy phân tích các quy định mới trong luật đường bộ"

    assert classify_query(question) == QueryClass.OPEN_ENDED


def test_direct_definition_policy_uses_low_expansion_retrieval():
    policy = retrieval_policy_for(QueryClass.DIRECT_DEFINITION, stream=False)

    assert policy.query_class == QueryClass.DIRECT_DEFINITION
    assert policy.retrieval_mode == "naive"
    assert policy.answer_mode == "hybrid"
    assert policy.top_k <= 10
    assert policy.chunk_top_k <= 5
    assert policy.final_top_n == 3
    assert policy.max_relation_tokens <= 1000
    assert policy.stream is False


def test_relational_policy_keeps_hybrid_retrieval():
    policy = retrieval_policy_for(QueryClass.RELATIONAL, stream=True)

    assert policy.query_class == QueryClass.RELATIONAL
    assert policy.retrieval_mode == "hybrid"
    assert policy.answer_mode == "hybrid"
    assert policy.final_top_n == 6
    assert policy.stream is True
