# Customer Support Ticket Classifier 

## 1. Project Info
- **Branch:** `feature/ticket-classifier`
- **Framework:** FastAPI

## 2. Dataset Summary
- **Source:** Simulated Customer Support Tickets
- **Raw Rows:** 570
- **Cleaned Rows:** 550 (After lowercase, trim, and duplicate removal)

## 3. Data Features & Labels
- **Input:** `ticket_text` (Support request text)
- **Target 1:** `category` (Technical Support, Billing & Invoice, Account Access, Feature Request, General Inquiry)
- **Target 2:** `priority` (Low, Medium, High, Urgent)

## 4. Stratified Data Splits (70 / 15 / 15)


- **Train Set (70%):** 378 rows
- **Validation Set (15%):** 71 rows
- **Test Set (15%):** 101 rows

## 5. Class Distribution

### Category Counts:
- Account Access: 120
- Feature Request: 114
- General Inquiry: 108
- Technical Support: 104
- Billing & Invoice: 104

### Priority Counts:
- Medium: 144
- Urgent: 142
- High: 139
- Low: 125