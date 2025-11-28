# Banking Conversation Intent Detection: Annotation Instructions (v2.0 Hybrid Action-Model)

## 1. Task Overview
*   **Objective:** Annotate each customer turn in banking conversations to train a **Hybrid NLU Model**.
*   **Core Philosophy:** Classify based on the **ACTION (The Verb)** the user intends to perform, NOT the **PRODUCT (The Noun)** they are discussing.
    *   The **Entity Model** (handled separately) will detect if the user said "Checking," "Mortgage," or "Debit Card."
    *   **Your Job:** Detect if the user wants to **Open**, **View**, **Pay**, **Fix**, or **Close** it.

**The "Verb Rule":**
*   *Old Way:* "User mentions 'Mortgage' -> Tag as `Investments`." (**INCORRECT**)
*   *New Way:* "User says 'Pay my mortgage'. Verb is 'Pay'. -> Tag as `Money Movement`." (**CORRECT**)

## 2. Annotation Output Format
For each customer turn, provide:
1.  `conversation_id`: Unique identifier.
2.  `turn_id`: Sequence number.
3.  `previous_intent`: The intent of the *previous* customer turn (context tracking).
4.  `current_intent`: The L1 category of the current action.
5.  `isboundary`: `true` if the user changes topic/action; `false` if continuing.
6.  `dialogueact`: The functional form of the utterance (request, question, etc.).

---

## 3. Intent Definitions (L1 Categories)

### 3.1 Accessibility & Language Support
**Definition:** Actions to configure the **communication interface** or request **physical/digital accommodations**. Focus is on *how* the user interacts with the bank.
*   **Core Verbs:** Change Language, Request Accommodation, Enable Accessibility.
*   **In-Scope Verbs/Actions:**
    *   Switching app language (e.g., English to Spanish).
    *   Requesting Braille or Large Print statements.
    *   Enabling screen reader support.
    *   Requesting TTY/TDD services.
*   **Verbatims (Examples):**
    *   "I need to switch the app to Spanish."
    *   "Do you have statements for the visually impaired?"
    *   "I need a sign language interpreter for my branch visit."
*   **Disambiguation:**
    *   "I can't hear the audio" (Technical failure) -> **Digital Banking Tech Support**.

### 3.2 Account Details
**Definition:** **READ-ONLY** inquiries about the **current state** or **attributes** of any financial product (Deposit or Loan). The user wants to *know* something, not *change* something.
*   **Core Verbs:** Check, Show, View, Find, What is.
*   **In-Scope Actions:**
    *   **Balances:** Checking balance of Checking, Savings, **Mortgages**, **Auto Loans**, Credit Cards.
    *   **Attributes:** Finding Routing Numbers, Account Numbers, SWIFT codes.
    *   **Status:** "Is my account active?", "Did my check clear?", "What is my **loan** interest rate?", "When is my **mortgage** payment due?"
*   **Verbatims (Examples):**
    *   "What is my checking account balance?"
    *   "How much do I owe on my car loan?" (**Note:** Loan Inquiry -> Account Details)
    *   "What is the routing number for my savings?"
    *   "Did the check I deposited yesterday clear?"
    *   "What is the interest rate on my mortgage?"
*   **Key Distinction (Read vs. Write):**
    *   "What is my overdraft limit?" -> **Account Details** (Read).
    *   "Increase my overdraft limit." -> **Account Management** (Write).

### 3.3 Account Management
**Definition:** **WRITE/STATE-CHANGE** actions regarding the **lifecycle** or **configuration** of a financial product. The user wants to alter the existence or settings of an account.
*   **Core Verbs:** Open, Close, Apply, Update, Add, Remove, Upgrade.
*   **In-Scope Actions:**
    *   **Origination:** Opening/Applying for Checking, Savings, Credit Cards, **Mortgages**, **Personal Loans**.
    *   **Termination:** Closing accounts, canceling services.
    *   **Configuration:** Adding joint owners, changing account types, setting up overdraft protection, refinancing a **loan**.
*   **Verbatims (Examples):**
    *   "I want to open a new savings account."
    *   "I need to apply for a personal loan." (**Note:** Loan Application -> Account Management)
    *   "Add my wife to my checking account."
    *   "Close my credit card account."
    *   "Refinance my mortgage."
*   **Disambiguation:**
    *   "Update my email address." (Profile change) -> **Profile Management**.
    *   "Change my PIN." (Credential change) -> **Card Maintenance**.

### 3.4 Bereavement & Estate Management
**Definition:** Administrative actions specifically triggered by the **death** of a customer.
*   **Core Verbs:** Report Death, Claim Assets, Manage Estate.
*   **Verbatims (Examples):**
    *   "My father passed away, what do I do with his account?"
    *   "Submit a death certificate."
    *   "I am the executor of this estate."

### 3.5 Money Movement
**Definition:** Actions that involve the **transfer of value** (funds) out of an account or between accounts.
*   **Core Verbs:** Pay, Transfer, Send, Schedule Payment.
*   **In-Scope Actions:**
    *   **Transfers:** Internal (Checking to Savings), External (P2P, Zelle, Wire).
    *   **Bill Pay:** Paying utilities, credit cards.
    *   **Loan Repayment:** "Pay my **mortgage**," "Make a payment on my **auto loan**."
*   **Verbatims (Examples):**
    *   "Transfer $50 to savings."
    *   "Pay my electric bill."
    *   "I want to pay my mortgage for this month." (**Note:** Loan Payment -> Money Movement)
    *   "Send money to Steve."
    *   "How do I pay my car loan?" (Procedural question regarding action -> Money Movement).
*   **Disambiguation:**
    *   "I want to deposit a check." (Money IN) -> **Deposits**.

### 3.6 Card Maintenance
**Definition:** Actions related to the **credential/instrument** (the plastic or virtual card), NOT the underlying financial account.
*   **Core Verbs:** Activate, Lock, Replace, Set PIN, Report Lost.
*   **In-Scope Actions:**
    *   Card Activation.
    *   Reporting Lost/Stolen/Damaged cards.
    *   Changing/Resetting PINs.
    *   Freezing/Unfreezing cards.
    *   Travel Notifications.
*   **Verbatims (Examples):**
    *   "I lost my debit card."
    *   "Activate my new credit card."
    *   "Change the PIN on my card."
    *   "Lock my card, I can't find it."
    *   "My card is cracked, send a new one."
*   **Disambiguation:**
    *   "Pay my credit card bill." (Money moving) -> **Money Movement**.
    *   "Increase my credit limit." (Account structure) -> **Account Management**.

### 3.7 Deposits
**Definition:** Actions related to moving funds **INTO** the bank from an external source.
*   **Core Verbs:** Deposit, Add Funds (Inbound).
*   **In-Scope Actions:** Mobile check deposit, ATM deposit inquiries, Direct deposit setup.
*   **Verbatims (Examples):**
    *   "I want to deposit a check with my camera."
    *   "Set up direct deposit for my paycheck."
    *   "Where can I deposit cash?"

### 3.8 Digital Banking Tech Support
**Definition:** Troubleshooting **technical barriers** to accessing or using the platform.
*   **Core Verbs:** Login, Reset Password, Crash, Error.
*   **Verbatims (Examples):**
    *   "I can't log in to the app."
    *   "Reset my password."
    *   "The app crashes when I click transfer."
    *   "My FaceID isn't working."

### 3.9 Disputes & Chargebacks
**Definition:** Contesting a **known/authorized** transaction due to merchant error or dissatisfaction.
*   **Core Verbs:** Dispute, Refund, Chargeback.
*   **Verbatims (Examples):**
    *   "I was charged twice for the same meal."
    *   "The merchant refused to refund me."
    *   "I cancelled this subscription but they still charged me."

### 3.10 Fees
**Definition:** Inquiries or complaints regarding **bank-levied charges**.
*   **Core Verbs:** Waive, Explain Fee.
*   **Verbatims (Examples):**
    *   "Why was I charged a monthly maintenance fee?"
    *   "Waive my overdraft fee please."
    *   "What is the fee for a wire transfer?"

### 3.11 International & FX
**Definition:** Actions specifically involving **foreign currency** or **cross-border** mechanics.
*   **Core Verbs:** Exchange, Convert.
*   **Verbatims (Examples):**
    *   "What is the exchange rate for the Euro?"
    *   "Can I use my card in Japan?"
    *   "Order foreign currency."

### 3.12 Investments
**Definition:** Actions related to **Wealth Management**, **Trading**, and **Securities**.
*   **Constraint:** **EXCLUDES** all Lending products (Mortgages, Loans).
*   **Core Verbs:** Trade, Buy, Sell (Securities), Analyze Portfolio.
*   **In-Scope Actions:** Stock trading, 401k/IRA management, Financial Advisor requests.
*   **Verbatims (Examples):**
    *   "Buy 10 shares of Apple."
    *   "How is my 401k performing?"
    *   "I want to speak to a financial advisor."
    *   "Sell my mutual funds."

### 3.13 Legal Orders & Government Requests
**Definition:** Compliance with **legal mandates**.
*   **Verbatims (Examples):**
    *   "I received a levy notice."
    *   "Why is my account garnished?"

### 3.14 Notifications & Alerts
**Definition:** Configuring **automated messaging**.
*   **Verbatims (Examples):**
    *   "Text me when my balance is low."
    *   "Stop sending me emails."

### 3.15 Profile Management
**Definition:** Updating **personal identity** and **contact info**.
*   **Verbatims (Examples):**
    *   "Update my home address."
    *   "I got married, change my last name."
    *   "Update my phone number."

### 3.16 Rewards & Offers
**Definition:** Loyalty programs and promotions.
*   **Verbatims (Examples):**
    *   "How many points do I have?"
    *   "Redeem my cash back."

### 3.17 Security & Fraud
**Definition:** Reporting **UNAUTHORIZED** activity, **Criminal** behavior, or **Identity Theft**.
*   **Core Verbs:** Report Fraud, Identity Theft, Hack.
*   **In-Scope Actions:** Unknown charges, Account Takeover, Phishing.
*   **Verbatims (Examples):**
    *   "I did not make this transaction."
    *   "Someone stole my identity."
    *   "I think my account was hacked."
*   **Key Distinction:**
    *   User knows the merchant but is angry? -> **Disputes**.
    *   User has no idea who the merchant is? -> **Security & Fraud**.

### 3.18 Statements & Documents
**Definition:** Retrieving **official records**.
*   **Verbatims (Examples):**
    *   "Download my statement."
    *   "I need a voided check."
    *   "Where is my 1099 tax form?"

### 3.19 Support & Locations
**Definition:** **General** information about the Bank, Branch Locations, and **Generic Product Policies**.
*   **Core Verbs:** Locate, Ask Policy.
*   **Verbatims (Examples):**
    *   "Where is the nearest ATM?"
    *   "What are your hours?"
    *   "Do you offer student loans?" (General Availability).
    *   "What is your policy on overdrafts?"
    *   "What credit cards do you offer?"
*   **Key Distinction (Generic vs. Specific):**
    *   "How do I pay my bill?" (Procedural Help) -> **Money Movement**.
    *   "What is your bill pay cutoff time?" (Policy Info) -> **Support & Locations**.

### 3.20 Transaction History & Details
**Definition:** **Searching** or **Identifying** past transactions (Read-Only).
*   **Core Verbs:** Search, List, Identify.
*   **Verbatims (Examples):**
    *   "Show me my last 5 transactions."
    *   "Did I spend money at Target last week?"
    *   "What is this pending charge?"

---

## 4. Disambiguation Rules & Logic Matrix

When an utterance is ambiguous, apply these priority rules.

### Rule 1: The Safety Priority (Security > All)
If the user implies theft, hacking, or danger, classifying as **Security & Fraud** takes precedence over everything.
*   *Utterance:* "My card was stolen and they drained my account."
*   *Legacy Risk:* Tagging as `Card Maintenance`.
*   *Correct Tag:* **Security & Fraud** (The crime is the priority).

### Rule 2: The "Read vs. Write" Distinction (Account Mgmt vs. Account Details)
Determine if the user wants to **CHANGE** state or **KNOW** state.
*   **CHANGE (Write):** Open, Close, Apply, Add User, Upgrade. -> **Account Management**.
*   **KNOW (Read):** Balance, Rate, Status, Routing Number. -> **Account Details**.

### Rule 3: The "Loan" Logic Flow
Correcting the "Investments" Error:
1.  **Paying it?** -> **Money Movement**.
2.  **Applying for it / Changing terms?** -> **Account Management**.
3.  **Checking balance / status?** -> **Account Details**.
4.  **Investing in stocks/bonds?** -> **Investments**.

### Rule 4: Procedural Questions (The "How-To" Rule)
If a user asks "How do I?", tag it as the Intent corresponding to that **ACTION**.
*   "How do I **pay** my bill?" -> **Money Movement**.
*   "How do I **change** my address?" -> **Profile Management**.
*   *Reasoning:* The user's goal is to perform the action. The bot should enter the flow for that action.

---

## 5. Dialogue Acts

Classify the **functional form** of the utterance.

| Label | Definition | Example |
| :--- | :--- | :--- |
| **request** | Explicit command or desire to act. | "Pay my bill." / "I want to pay." |
| **question** | Inquiry seeking info. | "Can I pay my bill?" / "What is my balance?" |
| **problem_statement** | Report of an issue. | "My payment failed." |
| **provide_information** | Supplying data. | "It's for the visa account." |
| **acknowledgment** | Phatic/Confirmation. | "Okay." / "Thanks." |

**Priority Hierarchy:** `request` > `problem_statement` > `question` > `provide_information` > `acknowledgment`.
