# uv_work

## 📅 목차

- [2025-09-04](#2025-09-04)
  
<br><br><br>

---

## 2025-09-04

### LangGraph + RAG 에이전트 구조

---

#### 🔹 1. 상태 정의 (`AgentState`)

```python
class AgentState(TypedDict):
    query: str            # 사용자의 질문
    context: List[Document]  # 검색된 관련 문서
    answer: str           # 최종 생성 답변
```

* 상태에는 질문, 문서, 답변이 포함됨
* 각 노드에서 상태를 입력받아 처리 후 업데이트

---

#### 🔹 2. 노드(Node) 정의

1. **retrieve**

   * 벡터 DB(Chroma)에서 질문과 관련된 문서 검색
   * 반환: `{'context': docs}`
2. **check\_doc\_relevence**

   * 검색된 문서가 질문과 관련 있는지 판단
   * 반환: `'generate'` 또는 `'rewrite'`
3. **generate**

   * RAG 체인을 통해 질문+문서를 기반으로 답변 생성
   * 반환: `{'answer': response}`
4. **rewrite**

   * 질문을 사전에 정의된 규칙/사전을 참고해 수정
   * 반환: `{'query': rewritten_query}`

---

#### 🔹 3. 조건부 엣지(Conditional Edge)

* **retrieve → generate / rewrite** 로 분기

```python
builder.add_conditional_edges('retrieve', check_doc_relevence)
```

* 관련 문서가 있으면 → `generate`
* 관련 문서가 부족하면 → `rewrite` → 다시 `retrieve`

---

#### 🔹 4. 그래프 연결

```python
builder.add_edge(START, 'retrieve')
builder.add_edge('rewrite', 'retrieve')
builder.add_edge('generate', END)
```

* START → retrieve → (조건부) generate/rewrite → END
* 문서 관련성이 낮으면 rewrite를 거쳐 retrieve를 다시 실행

---

#### 🔹 5. 그래프 실행

```python
initial_state = {"query": "연봉 5천만원 직장인의 소득세는?"}
response = graph.invoke(initial_state)
print(response['answer'].content)
```

* 사용자가 질문을 입력하면, LangGraph가 **RAG 기반 검색 → 관련성 체크 → 필요 시 질문 재작성 → 답변 생성** 흐름을 처리
* 상태(State)가 그래프 전체에서 계속 유지되므로 분기와 반복 처리가 자연스럽게 가능

---

#### 🔹 6. 핵심 포인트

1. **StateGraph**: 상태(State)를 기반으로 한 실행 흐름 정의
2. **Node**: 기능 단위(검색, 생성, 재작성)
3. **Conditional Edge**: 문서 관련성에 따라 분기 결정
4. **RAG 체인 + LLM**: 문서를 활용한 답변 생성
5. **재실행 구조**: rewrite 후 다시 retrieve 가능 → 유연한 질문 처리

---

📅[목차로 돌아가기](#-목차)

---
그림으로 보면 훨씬 직관적입니다. 원해?
