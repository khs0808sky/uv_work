# uv_work

## 📅 목차

- [2025-09-04](#2025-09-04)
- [2025-09-05](#2025-09-05)
- [2025-09-08](#2025-09-08)
  
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

## 2025-09-05

### LangGraph: 상태 관리 (State Management)

LangGraph에서 \*\*상태(State)\*\*는 그래프 실행 전체를 관통하며 노드 간 정보를 전달하는 핵심 구조입니다.
상태 관리가 잘 되어야 복잡한 RAG·에이전트 워크플로우도 안정적으로 동작할 수 있어요.

---

#### 🔹 상태(State)의 개념

* \*\*현재 그래프 실행의 맥락(Context)\*\*을 저장하는 **데이터 컨테이너**
* `TypedDict`나 Pydantic 모델 등으로 정의
* 각 노드가 입력받고, 처리한 값을 다시 상태에 업데이트
* 예시:

  ```python
  from typing import List, TypedDict
  from langchain_core.documents import Document

  class AgentState(TypedDict):
      query: str
      context: List[Document]
      answer: str
  ```

---

#### 🔹 상태(State) 관리 방식

1. **Immutable Update**

   * 노드 함수가 상태를 직접 수정하지 않고, **새로운 dict를 반환**
   * 각 노드가 필요한 부분만 추가/갱신

   ```python
   def retrieve(state: AgentState) -> AgentState:
       docs = retriever.invoke(state['query'])
       return {"context": docs}
   ```

   * 이렇게 하면 상태의 추적이 쉬움

2. **전체 상태 병합(Merge)**

   * LangGraph는 노드 반환값을 기존 상태와 병합
   * `state.update(node_return)` 자동 처리
   * 이 덕분에 필요한 값만 반환해도 상태가 유지됨

3. **단일 진입점 → 단일 종료점**

   * 그래프 시작 시 `initial_state`를 제공
   * END 노드까지 상태를 누적해 최종 결과 확인 가능

---

#### 🔹 상태 관리의 장점

* 복잡한 분기/루프에서도 **중간 결과 유지**
* LLM 호출, 검색 결과, 메타데이터 등을 한 객체에서 관리
* 디버깅과 로깅이 용이 (상태만 찍으면 전체 맥락 파악 가능)

---

### LangGraph: 라우팅 노드(Routing Node)와 함수(Function) 분리

LangGraph에서는 **로직을 라우팅 함수로 분리**해 흐름을 명확하게 구성할 수 있어요.

---

#### 🔹 라우팅 노드 개념

* **실제 데이터를 생성/가공하지 않고**, 단순히 **다음으로 이동할 노드 이름을 반환하는 함수**와 연결된 노드
* 예: 질문의 의도 분석 후 "검색 노드" 또는 "요약 노드"로 이동

---

#### 🔹 라우팅 함수 사용 예시

```python
def route(state: AgentState) -> str:
    # 질문이 "날씨" 관련이면 weather_node로 이동
    if "날씨" in state["query"]:
        return "weather_node"
    return "default_node"

builder.add_conditional_edges("router_node", route)
```

---

#### 🔹 함수와 노드 분리의 이점

1. **책임 분리 (Separation of Concerns)**

   * 로직 판단(`route`)과 데이터 처리(`generate`, `retrieve`)가 분리되어 유지보수가 쉬움
2. **그래프 구조가 명확해짐**

   * 각 노드의 역할이 단순화 → 디버깅, 재사용성 ↑
3. **확장성 높음**

   * 새로운 분기 로직 추가 시 라우터 함수만 수정하면 됨

---

#### 🔹 정리된 그래프 패턴

```
START → router_node → { weather_node | default_node } → END
```

* `router_node`는 상태만 보고 경로 결정
* 실제 데이터 생성은 `weather_node` 등에서 수행

---

📅[목차로 돌아가기](#-목차)

---

## 2025-09-08

### MCP (Multi-Choice Prompt / Multi-Choice Prediction)

MCP는 일반적으로 **LLM이나 AI 모델에서 여러 선택지를 기반으로 답변을 생성하거나 예측하는 기법**을 의미합니다.
문맥에 따라 정의가 조금 달라질 수 있지만, 기본 아이디어는 동일합니다.

---

#### 🔹 MCP의 핵심 개념

1. **다중 선택(Multi-Choice) 구조**

   * 모델에게 질문과 함께 여러 선택지를 제공
   * 모델이 답변으로 선택지 중 하나를 고르도록 유도
   * 예: "A, B, C 중 정답은 무엇인가요?"

2. **선택 기반 예측(Prediction)**

   * 단순 생성이 아닌 **정확한 선택**을 요구
   * 선택지마다 점수를 매기거나, 확률을 기반으로 최종 선택

3. **LLM에서 활용**

   * 다지선다형 퀴즈, 테스트, 분류 문제 등에 유용
   * 모델의 **논리적 추론** 능력과 **선택지 비교 능력**을 평가 가능
   * 예:

     ```
     질문: 지구의 위성은 무엇인가요? 
     선택지: A) 태양 B) 달 C) 화성
     MCP: B
     ```

---

#### 🔹 MCP 장점

* **정답 확인 가능**: 모델의 출력이 정량적/명확
* **논리적 비교 학습 가능**: 선택지 간 비교로 reasoning 학습
* **응용 범위 넓음**: 테스트, 분류, RAG/에이전트 결정 구조 등

---

#### 🔹 MCP 활용 예시

1. **RAG Agent에서 문서 선택**

   * 여러 문서 후보 중 가장 관련성 높은 문서를 선택
2. **분류 모델로 활용**

   * LLM에게 주어진 문장을 카테고리 A/B/C 중 선택하도록 함
3. **질문 재작성 판단**

   * 질문이 불명확하면 rewrite, 명확하면 generate 등 **조건부 라우팅**에 MCP 사용 가능

---

📅[목차로 돌아가기](#-목차)

---

