# DocInsight AI — Five-question test sheet

Use a text-based PDF with clear facts and visible page numbers. Upload and process it before asking the questions. The results below are from an executed two-page, in-memory PDF fixture (`Green_Campus_Demo.pdf`); verify again against the actual PDFs used in the demo.

| # | Test question | Expected behavior | Actual result / source | Pass? |
|---|---|---|---|---|
| 1 | What is the carbon emissions reduction target and deadline? | State only the supported target and deadline. | Retrieved 30% by 2030; `Green_Campus_Demo.pdf`, page 1 (score 0.491). | Pass |
| 2 | Which three improvements receive funding? | Give the three improvements from the cited passage. | Rooftop solar, classroom LED upgrades, and reusable refill stations; `Green_Campus_Demo.pdf`, page 1 (score 0.337). | Pass |
| 3 | Who proposed the initiative and how much initial funding was approved? | Report the proposer and amount from evidence. | Student Sustainability Council; $240,000 in March 2024; `Green_Campus_Demo.pdf`, page 1 (score 0.362). | Pass |
| 4 | Who publishes energy reports, and what happened in the library pilot? | Answer both details with the supporting source. | Maya Chen publishes every January; library pilot cut electricity use 18% over six months; `Green_Campus_Demo.pdf`, page 2 (score 0.505). | Pass |
| 5 | What is the college mascot's name? | Abstain: “I don't know based on the uploaded documents.” | No passage passed the 0.30 relevance threshold; the app returns the exact abstention response. | Pass |

## Manual workflow checks

- [x] Upload two PDFs and process them; observed 2 documents and 3 chunks.
- [x] Expand a source passage and verify its filename, page, and text against the fixture PDF.
- [x] Ask a supported question without a Gemini key and confirm **Extractive fallback**.
- [ ] If using a Gemini key, verify generated answer citations against the expandable passages (not tested; no key provided).
- [x] Ask an unrelated question and verify the exact abstention response.
- [x] Clear the chat and verify the conversation is removed.

The fixture test exercised PDF page extraction, semantic embeddings, FAISS indexing, top-passage retrieval, and low-relevance filtering. The Gemini network call was not tested because no API key was provided.
