# Banking Conversation Intent Detection - Annotation Prompt

## Task Overview
Annotate each customer turn in banking conversations with intent classification, boundary detection, and dialogue act labeling.

## Annotation Output Format
For each customer turn, provide:
- **intent**: What does the customer want?
- **isboundary**: Is it the start of a new topic? (true/false)
- **dialogueact**: How are they expressing it?
- **previous_intent**: Intent from the previous customer turn

---

## **CRITICAL: Human Judgment Required**

**DO NOT use code, regex, or keyword matching. This is a classification task requiring human-like reasoning.**

✅ Read full context and interpret meaning naturally  
✅ Consider what the customer actually means, not just surface words  
✅ Apply disambiguation rules thoughtfully  

❌ NO automated pattern matching, keyword scanning, or rigid if-then rules

**Example:** "I can't log in to pay bills"  
- ❌ Keyword: "pay bills" → Money Movement  
- ✅ Reasoning: Barrier is access → **Digital Banking Tech Support**

Annotate as an expert human annotator would.

---

## **Previous Intent Tracking Rule: 
When current_intent is "continuation":

The previous_intent should be the actual banking intent being continued (not "continuation")
Chain the intent forward through multiple continuation turns

Example:

Turn 0: current_intent = "Money Movement", previous_intent = "no_intent"
Turn 1: current_intent = "continuation", previous_intent = "Money Movement"
Turn 2: current_intent = "continuation", previous_intent = "Money Movement" (NOT "continuation")
Turn 3: current_intent = "Card Maintenance", previous_intent = "Money Movement"

Rule: previous_intent always refers to the last substantive banking intent from the previous customer turn, skipping over any "continuation" labels.

-----

## Intent Labels

### Meta Intents
- **no_intent**: Greetings, thanks (standalone/opening), casual talk, garbage text
- **continuation**: Continuing previous issue, acknowledging within same conversation flow
- **unknown**: Banking-related but genuinely ambiguous or doesn't fit taxonomy (use sparingly)

### Banking Intents

#### **Accessibility & Language Support**
Assistance related to language preferences, accessibility features, and accommodations for disabilities.
- Language preference changes
- Accessibility feature requests (large print, audio, screen reader support)
- ADA accommodations

---

#### **Account Details**
Informational inquiries about existing account specifics (NOT changes/actions).
- Balance inquiries
- Account number requests
- Routing/SWIFT codes
- Account status checks
- Interest rate on MY account
- "What's my account type?"
- Beneficiary information lookup

**NOT Account Details:**
- Changing account settings → Account Management
- Transaction lists → Transaction History & Details
- Problems with account access → Digital Banking Tech Support

---

#### **Account Management**
Opening, closing, or modifying account structure and services.
- Opening new accounts (checking, savings, credit cards, loans)
- Closing accounts
- Adding/removing account holders (joint accounts)
- Changing account type
- Adding/removing services (overdraft protection, auto-save, sweep accounts)
- Upgrading/downgrading accounts
- Account application status
- Zelle/Venmo/PayPal: Linking/unlinking accounts, enrollment, changing registered phone/email
- Setting up new products

**NOT Account Management:**
- Just asking about account balance/details → Account Details
- Personal info updates → Profile Management
- Making payments/transfers → Money Movement
- Alert preferences → Notifications & Alerts

**Disambiguation:**
- "Add overdraft protection" → Account Management
- "What's my overdraft limit?" → Account Details
- "Change my Zelle email" → Account Management (configuration)
- "My Zelle payment failed" → Money Movement (transaction issue)

---

#### **Bereavement & Estate Management**
Handling accounts after a customer's death or estate-related matters.
- Deceased account holder notifications
- Estate account management
- Death certificate submissions
- Beneficiary claims
- Trust account management related to estates

---

#### **Money Movement**
Transfers, payments, and moving money between accounts or to other parties.
- Internal transfers (between own accounts)
- External transfers (ACH, wire, P2P)
- Zelle/Venmo/PayPal payments and transfers
- Bill payments
- Payment scheduling/recurring payments
- Transfer status inquiries
- "My payment didn't go through"
- "How do I send money to someone?"
- Payment cancellations/modifications

**NOT Money Movement:**
- Incoming deposits from external sources → Deposits
- Zelle setup/enrollment → Account Management
- Technical issues preventing all banking functions → Digital Banking Tech Support

**Disambiguation:**
- "I can't log in to send money" → Digital Banking Tech Support (access barrier)
- "My Zelle transfer failed" → Money Movement (payment-specific issue)
- "How do I set up Zelle?" → Account Management (service setup)
- "I sent money to the wrong person" → Money Movement

---

#### **Card Maintenance**
Card-related services and routine card management.
- Card replacement (lost, stolen, damaged, expired)
- Card activation (after receiving replacement)
- PIN changes/reset
- Card freeze/unfreeze
- Travel notifications
- Card declined issues (non-fraud)
- ATM/POS transaction problems
- Virtual card numbers
- Card controls and spending limits
- Debit/credit card product inquiries

**NOT Card Maintenance:**
- First-time card request for brand new account → Account Management
- Fraudulent charges on card → Security & Fraud
- Disputing legitimate merchant charges → Disputes & Chargebacks

**Disambiguation:**
- "My card was stolen" (no fraud mentioned) → Card Maintenance
- "My card was stolen and someone made charges" → Security & Fraud
- "I need a replacement card" → Card Maintenance
- "I need a card for my new checking account" → Account Management

---

#### **Deposits**
Adding money to accounts from external sources (money coming IN).
- Mobile check deposits
- ATM deposits (cash/check)
- Direct deposit setup/issues
- Incoming wire transfers
- Paycheck deposit issues
- "My deposit isn't showing"
- Deposit holds/availability

**NOT Deposits:**
- Transfers between own accounts → Money Movement
- Sending money out → Money Movement

**Key Rule:** Money coming IN from outside = Deposits; Money moving OUT or BETWEEN accounts = Money Movement

---

#### **Digital Banking Tech Support**
Technical issues with online/mobile banking platforms that affect access or functionality.
- Login problems
- Password reset
- App crashes/freezes
- Browser compatibility
- Biometric authentication issues
- "Can't access my account online"
- Multi-factor authentication problems
- App not loading
- Technical error messages

**NOT Digital Banking Tech Support:**
- Issues specific to one function → Use that function's intent
- "Can't complete a transfer" → Money Movement
- "Can't see my statements" → Statements & Documents
- "Can't dispute a charge" → Disputes & Chargebacks

**Rule:** Cross-cutting technical barriers = Digital Banking Tech Support; Function-specific issues = That function's intent

---

#### **Disputes & Chargebacks**
Contesting authorized transactions or merchant-related issues.
- Merchant disputes (wrong amount, item not received, service not rendered)
- Duplicate charges
- Subscription cancellations not honored
- Return/refund not processed
- "I was charged twice"
- "Merchant charged wrong amount"

**NOT Disputes:**
- Fraudulent/unauthorized charges → Security & Fraud
- Bank fees → Fees
- Just asking about a transaction → Transaction History & Details

**Disambiguation:**
- Unknown/suspicious charges → Security & Fraud
- Known merchant but problematic transaction → Disputes & Chargebacks
- "I didn't authorize this" + recognizable merchant + customer may have forgotten → Disputes
- "I didn't authorize this" + completely unknown merchant → Security & Fraud

---

#### **Fees**
Questions about bank-imposed charges and fee-related requests.
- Monthly maintenance fees
- Overdraft fees
- ATM fees
- Wire transfer fees
- Foreign transaction fees
- "Why was I charged $X?"
- Fee waiver requests
- Fee schedule inquiries

**NOT Fees:**
- Merchant charges → Transaction History or Disputes
- Interest rates → Account Details

---

#### **International & FX**
Foreign transactions, currency exchange, and international banking.
- Currency exchange rates
- International wire transfers
- Foreign transaction issues
- Travel money/foreign currency
- Cross-border payment questions
- SWIFT transfers
- Foreign ATM usage

---

#### **Investments**
Investment products, lending products, and financial planning.
- Brokerage accounts
- Retirement accounts (IRA, 401k)
- CDs (Certificates of Deposit)
- **Loan products**: mortgages, personal loans, auto loans, home equity
- Investment advice
- Portfolio inquiries
- "What are your mortgage rates?"
- "Do you offer personal loans?"
- "What's the rate on a 5-year CD?"
- Loan applications and rate inquiries

---

#### **Legal Orders & Government Requests**
Compliance with legal mandates and government requests.
- Subpoenas
- Court orders
- Tax levies
- Garnishments
- Legal holds on accounts

---

#### **Notifications & Alerts**
Managing alert preferences and notification settings.
- Setting up balance alerts
- Transaction notifications
- Payment reminders
- Alert delivery method changes (text, email, push)
- "I'm not getting alerts"
- Modifying alert thresholds
- Enabling/disabling specific alerts

**NOT Notifications & Alerts:**
- Updating contact info for identity purposes → Profile Management
- Account structure changes → Account Management

**Disambiguation:**
- "Change my email for alerts" → Notifications & Alerts
- "Update my email address on file" → Profile Management

---

#### **Profile Management**
Updating personal identity and contact information.
- Name changes
- Address updates
- Phone number updates
- Email address updates (identity purposes)
- Employment information
- Communication preferences (paperless)
- Personal information corrections

**NOT Profile Management:**
- Account ownership changes → Account Management
- Alert delivery preferences → Notifications & Alerts

---

#### **Rewards & Offers**
Loyalty programs, promotions, and reward inquiries.
- Points/miles inquiries
- Cashback questions
- Redeeming rewards
- Special offers/promotions
- Rewards program enrollment
- Bonus qualification questions

---

#### **Security & Fraud**
Fraud prevention, security issues, and unauthorized access.
- Fraudulent charges by unknown parties
- Compromised credentials
- Identity theft concerns
- Account takeover
- Phishing/scam reports
- Security freezes
- Suspicious activity reports
- "Someone stole my card and used it"
- Password security concerns

**NOT Security & Fraud:**
- Merchant disputes → Disputes & Chargebacks
- Routine card replacement → Card Maintenance

**Key Distinction:**
- Unauthorized/criminal activity = Security & Fraud
- Authorized but problematic transaction = Disputes

---

#### **Statements & Documents**
Access to official bank documents and records.
- Monthly statements
- Tax documents (1099, 1098)
- Account opening documents
- Disclosure forms
- "I need my statement"
- Document download issues
- Paper vs. electronic statements

---

#### **Support & Locations**
General bank information, product offerings, and non-account-specific inquiries.
- Branch hours/locations
- ATM locations
- **Product offerings**: "What are your mortgage rates?", "What credit cards do you offer?", "Do you offer X service?"
- Eligibility questions: "What's the minimum age to open an account?"
- General interest rate inquiries (not tied to customer's account)
- Customer service contact information
- General procedural questions about bank policies

**NOT Support & Locations:**
- Routing/SWIFT codes → Account Details
- Customer's specific account questions → Appropriate specific intent
- Taking action to apply/open → Account Management

**Disambiguation:**
- "What are your mortgage rates?" → Support & Locations (general inquiry)
- "I want to apply for a mortgage" → Account Management (action)
- "What's my mortgage payment?" → Account Details
- "What credit cards do you offer?" → Support & Locations
- "I want to apply for your Rewards card" → Account Management

---

#### **Transaction History & Details**
Informational inquiries about past transactions (no action needed).
- "What did I buy at Target?"
- "Show me transactions from last month"
- "What's this $50 charge?" (pure information seeking)
- Transaction search/identification
- Categorization questions

**NOT Transaction History:**
- "This charge is wrong/fraudulent" → Disputes or Security & Fraud
- "Why was I charged a fee?" → Fees
- "Did my payment go through?" → Money Movement

**Rule:** Pure information seeking = Transaction History; Action needed = Appropriate specific intent

---

## Dialogue Acts

Classify HOW the customer is expressing their intent:

- **question**: Asking for information
- **problem_statement**: Describing an issue or problem
- **provide_information**: Giving information (account numbers, details, context)
- **acknowledgment**: Brief confirmations, agreements ("yes", "okay", "got it")
- **request**: Asking for action or service

### Priority Hierarchy (Select Only One)
When multiple dialogue acts could apply, use this priority:
**request > problem_statement > question > provide_information > acknowledgment**

### Examples:
- "Can you help me transfer money?" → **request** (not question)
- "My card isn't working" → **problem_statement**
- "What's my balance?" → **question**
- "My account number is 12345" → **provide_information**
- "Yes" → **acknowledgment**
- "I need a new card because mine was stolen" → **request** (has explicit ask)

---

## Disambiguation Rules

### Priority Hierarchy for Overlapping Intents:
1. **Security & Fraud** > all others (safety first)
2. **Disputes** > Transaction History (action > information)
3. **Money Movement** > Account Details (action > information)
4. **Problem-specific intent** > Digital Banking Tech Support (specific > general)
5. **Account Management** > Profile Management (structural > personal)

### Key Distinction Patterns:

**Direction of Money:**
- Money IN from outside → Deposits
- Money OUT or BETWEEN accounts → Money Movement

**Action vs. Information:**
- Seeking action → Functional intent
- Seeking information only → Details/History intents

**Specific vs. General Technical Issues:**
- One function affected → That function's intent
- Cross-cutting access/tech problem → Digital Banking Tech Support

**Authorization Context:**
- Unauthorized/criminal → Security & Fraud
- Authorized but problematic → Disputes & Chargebacks

**Configuration vs. Execution:**
- Setup/change settings → Account Management
- Execute transaction → Money Movement

**General Inquiry vs. Customer-Specific:**
- "What do you offer?" → Support & Locations
- "What do I have?" → Account Details
- "I want to get X" → Account Management

---

## Special Rules for Continuation and no_intent

### Continuation Rules:
- Use when customer is continuing the same topic from their previous turn
- "Yes", "No", "That's correct" in response to agent questions
- "Thanks" or acknowledgments that wrap up the current conversation
- Additional details on the same issue
- Takes `previous_intent` from the last **customer turn** (not agent turn)

### no_intent Rules:
- Standalone greetings: "Hi", "Hello", "Good morning"
- Standalone thanks (conversation opener or no prior context)
- Casual talk: "Have a nice day", "See you"
- Garbage/nonsense text
- Brief acknowledgments with no prior banking context

### Disambiguation: Thanks/Acknowledgments
- "Thanks" after agent resolves issue → **continuation**
- "Thanks" as conversation opener → **no_intent**
- "Thanks. Now about my card..." → New intent with `isboundary: true`

---

## Annotation Instructions

### Input Format:
TSV file where each line contains a JSON array representing one conversation broken down into turns.

### Output Format:
JSONL (JSON Lines) format where each line corresponds to the annotated JSON array for that conversation.

Each annotated object should include:
```json
{
  "conversation_id": <id>,
  "turn_id": <id>,
  "previous_intent": "<intent>",
  "current_intent": "<intent>",
  "isboundary": true/false,
  "dialogueact": "<act>"
}
```

### Annotation Rules:

1. **First turn**: `previous_intent` = "no_intent"
2. **Subsequent turns**: `previous_intent` = the intent from the previous **customer turn** (skip agent turns)
3. **isboundary**: 
   - `true` if starting a new intent (including first turn)
   - `true` if switching from one banking intent to another
   - `false` if continuation of the same intent
4. **Dialogue act**: Select only ONE based on priority hierarchy
5. **One JSON array per line**: Maintain array structure for each conversation
6. **Classification approach**: Use reasoning and context understanding, NOT regex or pattern matching. Interpret conversations naturally as a human would.

---

## Example

### Input:
```json
[
  {
    "conversation_id": 0,
    "turn_id": 0,
    "context": [],
    "current_turn": {
      "speaker": "Consumer",
      "text": "Our past president is still getting emails on this account. Can you please remove the email conkam2008@hotmail.com from all of our accounts"
    }
  },
  {
    "conversation_id": 0,
    "turn_id": 1,
    "context": [
      {"speaker": "Consumer", "text": "Our past president is still getting emails on this account. Can you please remove the email conkam2008@hotmail.com from all of our accounts"},
      {"speaker": "Agent", "text": "Is your inquiry regarding updating your email address?"}
    ],
    "current_turn": {
      "speaker": "Consumer",
      "text": "Yes"
    }
  }
]
```

### Expected Output:
```json
[
  {
    "conversation_id": 0,
    "turn_id": 0,
    "previous_intent": "no_intent",
    "current_intent": "Profile Management",
    "isboundary": true,
    "dialogueact": "request"
  },
  {
    "conversation_id": 0,
    "turn_id": 1,
    "previous_intent": "Profile Management",
    "current_intent": "continuation",
    "isboundary": false,
    "dialogueact": "acknowledgment"
  }
]
```

---

## Additional Examples for Clarity

| Message | Previous Intent | Current Intent | isboundary | dialogueact | Reasoning |
|---------|----------------|----------------|------------|-------------|-----------|
| "Hello, I need help" | no_intent | unknown | true | request | First turn, vague banking request |
| "My Zelle isn't working" | no_intent | Money Movement | true | problem_statement | Payment function issue |
| "Can you change my Zelle email?" | no_intent | Account Management | true | request | Configuration change |
| "Thanks!" | Money Movement | continuation | false | acknowledgment | Wrapping up transfer help |
| "What's my routing number?" | no_intent | Account Details | true | question | Account information |
| "I can't log in" | no_intent | Digital Banking Tech Support | true | problem_statement | Access barrier |
| "What are your mortgage rates?" | no_intent | Support & Locations | true | question | General product inquiry |
| "I want to apply for a mortgage" | Support & Locations | Account Management | true | request | Taking action to open product |
| "Yes, that's correct" | Card Maintenance | continuation | false | acknowledgment | Continuing card issue |
| "This charge is fraudulent" | no_intent | Security & Fraud | true | problem_statement | Unauthorized activity |
| "This charge is wrong" | no_intent | Disputes & Chargebacks | true | problem_statement | Authorized but problematic |

---

## Critical Reminders

1. **Context is key**: Always consider the full conversation context, not just isolated messages
2. **Use unknown sparingly**: Only when genuinely ambiguous after considering all context
3. **One dialogue act only**: Apply priority hierarchy strictly
4. **Previous intent tracks customer turns**: Skip agent messages when determining previous_intent
5. **Human judgment required**: Use natural interpretation, NOT code or pattern matching
6. **Boundary detection**: New topic = true, continuation = false
7. **When in doubt**: Refer to disambiguation rules and priority hierarchies
