# RAG Evaluation Guide: Traffic Law Assistant

This document contains specialized test cases designed to evaluate the performance difference between **Naive RAG** (standard vector search) and **Hybrid GraphRAG** (graph-linked retrieval).

## Evaluation Methodology
To truly see the difference, ask questions that require "connecting the dots" across different sections of the laws (35/2024/QH15 - Trật tự, an toàn giao thông đường bộ and 36/2024/QH15 - Đường bộ).

---

## 1. The "Definition vs. Application" Gap
**Question:** "Người điều khiển phương tiện tham gia giao thông đường bộ được định nghĩa như thế nào và họ có những trách nhiệm gì khi gặp sự cố trên đường cao tốc?"

**Why compare:** Naive RAG will likely find the definition in one chunk and the responsibilities in another, but it might struggle to emphasize how the specific definition of the person links to the legal duties defined separately. Hybrid GraphRAG should link the entity "Người điều khiển" across multiple articles.

## 2. Cross-Regulation Penalties (The "Connection" Test)
**Question:** "Phân biệt phạm vi điều chỉnh giữa Luật Trật tự, an toàn giao thông đường bộ và Luật Đường bộ. Có những hành vi nào bị cấm chung trong cả hai luật này không?"

**Why compare:** This requires summarizing two entire documents and finding overlapping "Hành vi bị cấm" (Prohibited acts). Naive RAG often misses the global overlap, while Hybrid mode uses the graph to see nodes that appear in both contexts.

## 3. Responsibility Hierarchy
**Question:** "Cơ quan nào chịu trách nhiệm chính về quản lý nhà nước đối với đường bộ và sự phối hợp giữa Bộ Công an với Bộ Giao thông vận tải được quy định ra sao?"

**Why compare:** Responsibilities are often scattered throughout legal texts. GraphRAG excels at pulling all edges connected to the "Bộ Công an" and "Bộ Giao thông vận tải" nodes to show the relationship (coordination) rather than just keyword matches.

## 4. Technical Requirements for Specific Infrastructure
**Question:** "Điều kiện để đưa đường cao tốc vào khai thác là gì và các quy định này liên quan thế nào đến an toàn giao thông?"

**Why compare:** The "technical conditions" are usually in the Road Law ( Luật Đường bộ), but their "safety purpose" is in the Traffic Safety Law. Hybrid RAG can traverse the graph from "Đường cao tốc" to "Điều kiện khai thác" to "An toàn giao thông".

## 5. Multi-Step Situational Analysis
**Question:** "Khi xảy ra tai nạn giao thông trên đường bộ, quy trình xử lý và trách nhiệm của các bên liên quan (người điều khiển, cơ quan chức năng, người có mặt tại hiện trường) được quy định như thế nào?"

**Why compare:** This is a classic "global" query. Naive RAG will give you snippets of Article X or Article Y. Hybrid RAG will create a more cohesive "narrative" of the whole process by following the sequence of entities involved in the event.

---
**Pro-tip:** Enable **Comparison Mode** in the UI and look for responses where Hybrid GraphRAG mentions specific relationships (e.g., "theo mối liên hệ với...", "được quy định phối hợp bởi...")—this is the Knowledge Graph at work!
