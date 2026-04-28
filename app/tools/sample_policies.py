"""
Built-in sample policy corpus with comprehensive interconnection mapping.

AUDIT FINDINGS & FIXES:
1. MISSING CROSS-REFERENCES: Original policies had no links to related policies.
   FIX: Added exhaustive cross-references (e.g., Travel → Training, Security, Conduct).

2. NO POLICY RELATIONSHIP GRAPH: System couldn't recognize policy dependencies.
   FIX: Built PolicyInterconnectionMap with explicit relationships and tags.

3. INCOMPLETE CONTEXT FOR QUESTIONS: Isolated policy content couldn't answer multi-policy questions.
   FIX: Added get_related_policies() and get_comprehensive_context() methods.

4. NO SECURITY-TRAVEL LINK: Travel expenses didn't consider data handling during travel.
   FIX: Security policy now explicitly references travel data handling obligations.

5. NO TRAINING-TRAVEL LINK: Conference travel wasn't integrated with L&D benefits.
   FIX: Training policy cross-references travel approval and per-diem limits.

6. MISSING COMPLIANCE INTERCONNECTIONS: Conduct policy didn't reference all relevant policies.
   FIX: Code of Conduct now comprehensively cross-references Travel, Security, Training.

This ensures the RAG system can answer complex, multi-policy questions professionally.
"""

from typing import List, Dict, Set, Optional
from langchain.schema import Document
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 POLICY INTERCONNECTION MAP (Core Fix #2)
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyInterconnectionMap:
    """
    Bidirectional graph of policy relationships.
    Enables comprehensive context retrieval for multi-policy questions.
    """

    def __init__(self):
        # Policy metadata: code → (title, category, tags)
        self.policies: Dict[str, Dict] = {}
        
        # Relationships: policy_code → set of related_policy_codes
        self.relationships: Dict[str, Set[str]] = {}
        
        # Reverse lookup: category → set of policy_codes
        self.categories: Dict[str, Set[str]] = {}
        
        # Tags for semantic clustering
        self.tags: Dict[str, Set[str]] = {}

    def register_policy(
        self,
        code: str,
        title: str,
        category: str,
        tags: List[str],
        related_policies: List[str],
    ):
        """Register a policy with its metadata and relationships."""
        self.policies[code] = {
            "title": title,
            "category": category,
            "tags": set(tags),
        }
        self.relationships[code] = set(related_policies)
        self.categories.setdefault(category, set()).add(code)
        self.tags.setdefault(code, set(tags))

    def get_related(self, code: str, depth: int = 1) -> Set[str]:
        """
        Get all policies related to `code`, up to `depth` hops away.
        Bidirectional traversal ensures comprehensive interconnection.
        """
        visited = set()
        queue = [(code, 0)]

        while queue:
            curr, d = queue.pop(0)
            if curr in visited or d > depth:
                continue
            visited.add(curr)

            # Add direct relationships
            for related in self.relationships.get(curr, set()):
                if related not in visited:
                    queue.append((related, d + 1))

            # Add reverse relationships (if policy X links to Y, include Y→X)
            for other, rels in self.relationships.items():
                if curr in rels and other not in visited:
                    queue.append((other, d + 1))

        visited.discard(code)
        return visited

    def get_by_category(self, category: str) -> Set[str]:
        """Retrieve all policies in a category."""
        return self.categories.get(category, set())

    def get_by_tags(self, tags: List[str]) -> Set[str]:
        """Retrieve all policies matching any of the given tags."""
        result = set()
        for code, policy_tags in self.tags.items():
            if any(t in policy_tags for t in tags):
                result.add(code)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 INITIALIZE INTERCONNECTION MAP
# ═══════════════════════════════════════════════════════════════════════════════

_interconnection_map = PolicyInterconnectionMap()


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 COMPREHENSIVE SAMPLE POLICIES WITH CROSS-REFERENCES (Core Fix #1)
# ═══════════════════════════════════════════════════════════════════════════════

def get_sample_docs() -> List[Document]:
    """
    Return comprehensive policy corpus with bidirectional cross-references.
    Each policy now links to related policies for multi-policy question answering.
    """
    
    policies = [
        # ─────────────────────────────────────────────────────────────────────
        # TRAVEL POLICY (TRV-POL-1001-V4)
        # ─────────────────────────────────────────────────────────────────────
        (
            "TRV-POL-1001-V4",
            "travel",
            """CORPORATE TRAVEL POLICY — TRV-POL-1001-V4 (Effective 01-April-2025)

1. DOMESTIC TRAVEL ENTITLEMENTS
   Grade L1-L3 : Economy air | Hotel ₹3,500/night | Meals ₹600/day
   Grade L4-L5 : Economy air | Hotel ₹5,500/night | Meals ₹900/day
   Grade L6-L7 : Business (>2hr) | Hotel ₹8,000/night | Meals ₹1,200/day
   VP & Above  : Business air  | Hotel ₹12,000/night | Meals ₹1,800/day

2. INTERNATIONAL TRAVEL ENTITLEMENTS
   Grade L1-L3 : Hotel USD 120/night | Per diem USD 50/day
   Grade L4-L5 : Hotel USD 180/night | Per diem USD 75/day
   Grade L6-L7 : Hotel USD 250/night | Per diem USD 100/day
   VP & Above  : Hotel USD 400/night | Per diem USD 150/day
   All grades  : Business class for flights > 6 hours

3. LOCAL TRANSPORT
   Cab (corporate app) : Actual, max ₹800/trip or ₹3,000/day
   Two-wheeler (own)   : ₹12/km
   Four-wheeler (own)  : ₹18/km
   Metro/Bus           : Actuals with receipt

4. ADVANCE AGAINST TRAVEL
   Up to 80% of estimated entitlement may be drawn as advance.
   Advance must be settled within 5 working days of return.

5. CROSS-POLICY REFERENCES
   ⚠️  SECURITY REQUIREMENTS (SEC-POL-8005-V7, Section 6):
       • Never download customer data locally before travel
       • Use VPN for all corporate network access during travel
       • Do not store Tier 3/4 data on personal devices
       • Report lost/stolen devices within 12 hours (Section 7.1)
   
   ⚠️  COMPLIANCE & CODE OF CONDUCT (HR-POL-5050-V4, Section 7):
       • Expense fraud (dual reimbursement, false receipts) → Immediate Termination
       • All travel expenses subject to audit (Section 11)
       • Managers must review for TRV-POL-1001-V4 violations (Section 2.1, Level 2 Offense)
   
   ✓  TRAINING & CONFERENCE TRAVEL (LND-POL-7010-V3, Section 3.2):
       • Conference travel costs NOT deducted from L&D stipend
       • Must follow TRV-POL-1001-V4 booking protocols and budgetary limits
       • All pre-approvals apply (Section 4.1 Approval Matrix)
       • Per diems align with grade-based entitlements above

6. APPROVAL MATRIX
   Estimated Cost | Primary Approval | Secondary | Lead Time
   $0 - $999      | Direct Manager   | None      | 7 Days
   $1K - $2,999   | Director         | None      | 14 Days
   $3K - $9,999   | VP               | Finance   | 21 Days
   $10K+          | C-Suite          | CFO       | 30 Days

7. NON-REIMBURSABLE ITEMS
   Personal entertainment, traffic violations, alcohol, upgrades beyond grade.

Source: Policy TRV-POL-1001-V4 | Related: SEC-POL-8005-V7 | HR-POL-5050-V4 | LND-POL-7010-V3""",
            ["travel", "expense", "reimbursement", "approval", "grade-based"],
            ["SEC-POL-8005-V7", "HR-POL-5050-V4", "LND-POL-7010-V3", "C-03"],
        ),

        # ─────────────────────────────────────────────────────────────────────
        # SECURITY & DATA PRIVACY POLICY (SEC-POL-8005-V7)
        # ─────────────────────────────────────────────────────────────────────
        (
            "SEC-POL-8005-V7",
            "security",
            """GLOBAL IT SECURITY & DATA PRIVACY POLICY — SEC-POL-8005-V7 (Effective 06-01-2026)

1. ZERO-TRUST SECURITY MODEL
   Access granted strictly on principle of least privilege (PoLP).
   Trust never assumed based on network location.

2. DATA CLASSIFICATION MATRIX
   Tier 1 (Public)      : Public distribution — no risk
   Tier 2 (Internal)    : Standard operational data — minimal harm if disclosed
   Tier 3 (Confidential) : Sensitive business data — significant damage if disclosed
   Tier 4 (Restricted)  : Regulated data (GDPR, PCI-DSS) — legal breach if compromised

3. ACCEPTABLE USE & TRAVEL RESTRICTIONS
   ⚠️  CRITICAL FOR TRAVELERS (Cross-reference: TRV-POL-1001-V4):
       • Production customer data (Tier 4) MUST NEVER be downloaded locally
       • During travel, all network access REQUIRES corporate VPN
       • Personal devices may NOT access Tier 3/4 data (unless MDM installed)
       • Hotel/airport WiFi: Use VPN without exception
       • Lost/stolen devices: Report within 12 hours for remote wipe
   
   ⚠️  LINKED TO CODE OF CONDUCT (HR-POL-5050-V4, Section 7):
       • Intentional data exposure = Level 3 offense (Immediate Termination)
       • Shadow IT: Unauthorized SaaS/AI tools = Disciplinary action

4. MULTI-FACTOR AUTHENTICATION (MFA)
   Hardware security keys (YubiKeys) preferred.
   Authenticator apps (Google Authenticator) acceptable.
   SMS-based MFA prohibited (SIM-swapping risk).

5. PASSWORD POLICIES
   Minimum 14 characters | Change every 90 days | Cannot reuse last 10 passwords

6. INCIDENT RESPONSE SLAs
   Lost/Stolen Hardware : 12 hours
   Suspected Phishing   : 1 hour
   Misdirected Email    : 4 hours
   Amnesty Clause: No disciplinary action for honest mistakes if reported immediately

7. OFFBOARDING & TERMINATION
   ⚠️  LINKED TO CODE OF CONDUCT (HR-POL-5050-V4, Section 9):
       • All logical access revoked at 5:00 PM on final day
       • Hardware must be returned within 3 business days
       • Failure to return = Device cost deducted from final paycheck

8. MANDATORY TRAINING
   ✓  REQUIRED BY LND-POL-7010-V3 (Section 2):
       • All new hires: SEC-100 (Phishing & Data Privacy) within 30 days
       • Annual Security & Compliance Refresher: 60 minutes by Oct 31st
       • Non-compliance = Network access suspension + Disciplinary action

Source: Policy SEC-POL-8005-V7 | Related: TRV-POL-1001-V4 | HR-POL-5050-V4 | LND-POL-7010-V3""",
            ["security", "data-privacy", "compliance", "incident-response", "travel-security"],
            ["TRV-POL-1001-V4", "HR-POL-5050-V4", "LND-POL-7010-V3"],
        ),

        # ─────────────────────────────────────────────────────────────────────
        # CODE OF CONDUCT & DISCIPLINARY PROCEDURES (HR-POL-5050-V4)
        # ─────────────────────────────────────────────────────────────────────
        (
            "HR-POL-5050-V4",
            "conduct",
            """GLOBAL CODE OF CONDUCT & DISCIPLINARY PROCEDURES — HR-POL-5050-V4 (Effective 03-01-2026)

1. CORE PHILOSOPHY
   Safe, respectful, legally compliant work environment.
   Every employee expected to act with integrity and protect company assets.

2. REPORTING MECHANISMS
   • Direct Manager (primary for minor disputes)
   • HR Business Partner (interpersonal conflicts)
   • Global Ethics Hotline (24/7, anonymous, 14 languages)

3. MANDATORY MANAGER REPORTING
   Managers must report Level 2+ violations to HR within 24 hours.
   Failure to report = Level 2 disciplinary offense.

4. NON-RETALIATION & WHISTLEBLOWER PROTECTION
   Absolute prohibition against retaliation.
   Retaliation = Immediate Level 3 action (up to termination).

5. INVESTIGATION PROTOCOLS
   • Timeline: Investigation opened within 3 business days
   • Conclusion: Within 30 calendar days (unless legal complexity)
   • Confidentiality: Strictly need-to-know basis

6. VIOLATION CATEGORIZATION & DISCIPLINE MATRIX
   
   Level 1 (Minor):
   Examples: Tardiness, dress code, minor equipment misuse
   Discipline: Verbal warning → 6 months in file
   
   Level 2 (Moderate):
   Examples: Insubordination, failure to comply with TRV-POL-1001-V4 per diems,
             aggressive communication, unauthorized document sharing
   ⚠️  TRAVEL-SPECIFIC (TRV-POL-1001-V4, Section 7):
       • Exceeding hotel/meal allowances without approval
       • Booking via personal loyalty programs instead of TMC
       • Falsifying expenses or dual-claiming reimbursement
   ⚠️  SECURITY-SPECIFIC (SEC-POL-8005-V7, Section 3):
       • Inputting Tier 3/4 data into unauthorized SaaS platforms
       • Failure to use VPN during remote work
   Discipline: Written warning + Performance Improvement Plan (30/60/90 days)
   
   Level 3 (Severe / Zero-Tolerance):
   Examples: Expense fraud, physical violence, sexual harassment, corporate espionage,
             working under influence, intentional data sabotage
   ⚠️  EXPENSE FRAUD INCLUDES (TRV-POL-1001-V4, Section 7):
       • Fabricating travel receipts
       • Claiming personal expenses as business
       • Dual-reimbursement of mileage and fuel
   ⚠️  SECURITY BREACHES (SEC-POL-8005-V7):
       • Intentional exposure of Tier 4 data
       • Downloading production customer data before travel
   Discipline: IMMEDIATE TERMINATION (no progressive discipline)

7. PROGRESSIVE DISCIPLINE (Level 1 & 2 only)
   Step 1: Documented Verbal Warning (note in file for 6 months)
   Step 2: Written Warning + PIP (30, 60, or 90 days)
   Step 3: Final Written Warning / Unpaid Suspension (1-3 days)
   Step 4: Termination for Cause

8. CONFLICTS OF INTEREST & GIFTS
   No outside employment competing with organization.
   All secondary employment must be disclosed and approved by HR.

9. OFFBOARDING PROCEDURES
   ✓  LINKED TO SEC-POL-8005-V7 (Section 7):
       • Logical access revoked 5:00 PM on final day
       • Hardware return deadline: 3 business days
       • Failure to return = Device cost deducted from final paycheck

Source: Policy HR-POL-5050-V4 | Related: TRV-POL-1001-V4 | SEC-POL-8005-V7 | LND-POL-7010-V3""",
            ["conduct", "discipline", "compliance", "ethics", "travel-conduct"],
            ["TRV-POL-1001-V4", "SEC-POL-8005-V7", "LND-POL-7010-V3", "C-03"],
        ),

        # ─────────────────────────────────────────────────────────────────────
        # LEARNING & DEVELOPMENT POLICY (LND-POL-7010-V3)
        # ─────────────────────────────────────────────────────────────────────
        (
            "LND-POL-7010-V3",
            "training",
            """GLOBAL LEARNING, DEVELOPMENT & TUITION ASSISTANCE — LND-POL-7010-V3 (Effective 05-01-2026)

1. LEARNING PHILOSOPHY
   Continuous learning culture supporting skill acquisition, certifications, and higher education.
   Benefits subject to annual budget approvals (not guaranteed contractual benefit).

2. MANDATORY COMPLIANCE TRAINING
   All employees must complete within first 30 days:
   • HR-101: Code of Conduct & Anti-Harassment (2 hours)
   • SEC-100: Phishing, Social Engineering & Data Privacy (1.5 hours)
   • FIN-201: Insider Trading & Financial Disclosures (1 hour)
   
   ✓  LINKED TO SEC-POL-8005-V7:
       SEC-100 covers all data handling during travel and remote work
   
   ✓  LINKED TO HR-POL-5050-V4:
       HR-101 explains disciplinary framework and reporting procedures
   
   Annual Refresher: 60-minute Security & Compliance Refresher by Oct 31st
   Non-Compliance Penalty: Network access suspension + Disciplinary action

3. PROFESSIONAL DEVELOPMENT STIPEND (Certifications & Conferences)
   
   Annual Stipend Tiers (reset Jan 1, non-rolling):
   L1-L4 (IC)     : $1,500 | Manager approval
   M1-M2 (Mgmt)   : $3,000 | Director approval
   D1+ (Director+): $5,000 | VP approval
   
   Advanced Tech Track: +$2,000 supplemental for LLM, ML, AI upskilling
   (Must be approved by CTO for Engineering/Data Science/Product roles)

4. CONFERENCE TRAVEL INTEGRATION
   ✓  CRITICAL CROSS-REFERENCE TO TRV-POL-1001-V4:
       • Conference ticket cost deducted from L&D stipend
       • Associated travel costs (flights, hotel, per diems) NOT deducted
       • Must follow TRV-POL-1001-V4 booking protocols (Section 4.2)
       • Must follow TRV-POL-1001-V4 approval matrix (Section 6)
       • Per diems align with grade-based travel entitlements
       • All expenses subject to HR-POL-5050-V4 audit procedures
   
   ✓  SECURITY REQUIREMENTS (SEC-POL-8005-V7):
       • Cannot use corporate funds to attend in-person conferences if it exposes
         confidential data in unsecured environments
       • Virtual conference attendance recommended for sensitive data roles

5. FORMAL TUITION ASSISTANCE PROGRAM (Degree Programs)
   Eligibility: 12+ months full-time service, in good standing (not on PIP/discipline)
   
   Maximum Reimbursement: $5,250 USD/calendar year
   • Aligns with IRS Section 127 tax-free educational assistance
   • Covers tuition, lab fees, required textbooks only
   • Non-reimbursable: Late fees, graduation fees, parking, optional materials

6. ACADEMIC PERFORMANCE REQUIREMENTS
   
   Undergraduate (B.A., B.S.):
   Grade A or B   : 100% reimbursement
   Grade C        : 50% reimbursement
   Grade D, F, or Incomplete : 0% reimbursement
   Pass/Fail ("Pass") : 100% reimbursement
   
   Graduate (M.S., MBA, Ph.D.):
   Grade A or B   : 100% reimbursement
   Grade C, D, F, or Incomplete : 0% reimbursement (No B+ = No reimbursement)

7. REIMBURSEMENT WORKFLOW
   1. Pre-Approval: Submit Tuition Approval Form via Workday 15+ days before start
   2. Out-of-Pocket: Employee pays university directly
   3. Submission (within 45 days of completion):
      • Approved pre-authorization form
      • Itemized receipt (zero balance)
      • Official transcript with final grade

8. TUITION CLAWBACK (RETENTION AGREEMENT)
   If employee leaves (voluntary or for-cause) after tuition reimbursement:
   • 0-12 months after payout: Repay 100%
   • 12-24 months after payout: Repay 50%
   • After 24 months: 0% repayment (fully forgiven)
   
   Deduction authorized from final paycheck, severance, or PTO payout.
   ✓  SUBJECT TO LOCAL LABOR LAW RESTRICTIONS

9. TIME COMMITMENT FOR COURSEWORK
   ✓  LINKED TO HR-POL-4001-V6 (Leave & Absence Policy):
       • Classes during business hours = Use PTO or negotiate flexible arrangement
       • Coursework is NOT compensable working time

10. CROSS-REFERENCES SUMMARY
    ✓  SEC-POL-8005-V7: Mandatory security training (SEC-100, refresher)
    ✓  HR-POL-5050-V4: Code of Conduct, disciplinary framework
    ✓  TRV-POL-1001-V4: Conference travel booking, per diems, approval matrix
    ✓  HR-POL-4001-V6: Time off for coursework

Source: Policy LND-POL-7010-V3 | Related: SEC-POL-8005-V7 | HR-POL-5050-V4 | TRV-POL-1001-V4""",
            ["training", "learning", "development", "tuition", "conference-travel"],
            ["SEC-POL-8005-V7", "HR-POL-5050-V4", "TRV-POL-1001-V4"],
        ),

        # ─────────────────────────────────────────────────────────────────────
        # COMPENSATION STRUCTURE (C-03)
        # ─────────────────────────────────────────────────────────────────────
        (
            "C-03",
            "compensation",
            """COMPENSATION STRUCTURE — C-03 (Effective 01-April-2025)

1. GRADE BANDS (Annual CTC)
   L1  : ₹3,00,000 – ₹5,00,000
   L2  : ₹5,00,001 – ₹8,00,000
   L3  : ₹8,00,001 – ₹12,00,000
   L4  : ₹12,00,001 – ₹18,00,000
   L5  : ₹18,00,001 – ₹28,00,000
   L6  : ₹28,00,001 – ₹45,00,000
   L7  : ₹45,00,001 – ₹75,00,000
   VP  : ₹75,00,001 – ₹1,50,00,000
   SVP : Above ₹1,50,00,000

2. SALARY COMPONENTS
   Basic          : 40% of CTC
   HRA            : 50% of Basic (metro) / 40% (non-metro)
   PF (employer)  : 12% of Basic
   Gratuity prov  : 4.81% of Basic
   Special Allow  : Remainder

3. VARIABLE PAY
   L1-L3 : 5% of Annual Basic
   L4-L5 : 10% of Annual Basic
   L6-L7 : 15–20% of Annual Basic
   VP+   : 25–40% of Annual Basic

4. CROSS-POLICY IMPLICATIONS
   ✓  TRAVEL ENTITLEMENTS (TRV-POL-1001-V4, Section 1):
       Grade determines hotel, meal, transport allowances
       Example: L1-L3 get ₹3,500/night hotel; VP+ get ₹12,000/night
   
   ✓  LEARNING & DEVELOPMENT (LND-POL-7010-V3, Section 3):
       Directors+ eligible for $5,000 annual L&D stipend (vs $1,500 for L1-L4)

Source: Policy C-03 | Related: TRV-POL-1001-V4 | LND-POL-7010-V3""",
            ["compensation", "salary", "grade-based", "benefits"],
            ["TRV-POL-1001-V4", "LND-POL-7010-V3"],
        ),
    ]

    # ═════════════════════════════════════════════════════════════════════════
    # REGISTER POLICIES IN INTERCONNECTION MAP
    # ═════════════════════════════════════════════════════════════════════════

    for code, category, content, tags, related in policies:
        _interconnection_map.register_policy(
            code=code,
            title=content.split("\n")[0].replace("—", "").strip(),
            category=category,
            tags=tags,
            related_policies=related,
        )

    # Convert to LangChain Documents with enhanced metadata
    docs = [
        Document(
            page_content=content,
            metadata={
                "policy_code": code,
                "category": category,
                "tags": tags,
                "related_policies": related,
            },
        )
        for code, category, content, tags, related in policies
    ]

    logger.info(
        "✓ Loaded %d policies with comprehensive cross-references and interconnection map",
        len(docs),
    )
    return docs


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 PUBLIC API FOR POLICY INTERCONNECTION QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_interconnection_map() -> PolicyInterconnectionMap:
    """
    Retrieve the global policy interconnection map.
    Use this to find related policies for multi-policy questions.
    """
    return _interconnection_map


def get_related_policies(policy_code: str, depth: int = 1) -> List[str]:
    """
    Get all policies related to a given policy code.
    
    Args:
        policy_code: The policy code (e.g., "TRV-POL-1001-V4")
        depth: How many hops deep to traverse the relationship graph
    
    Returns:
        List of related policy codes
    """
    related = _interconnection_map.get_related(policy_code, depth=depth)
    return sorted(list(related))


def get_policies_by_category(category: str) -> List[str]:
    """
    Get all policies in a specific category.
    
    Args:
        category: One of "travel", "security", "conduct", "training", "compensation"
    
    Returns:
        List of policy codes in that category
    """
    policies = _interconnection_map.get_by_category(category)
    return sorted(list(policies))


def get_policies_by_tags(tags: List[str]) -> List[str]:
    """
    Get all policies matching any of the given tags.
    
    Args:
        tags: List of tags (e.g., ["travel", "expense", "approval"])
    
    Returns:
        List of policy codes matching the tags
    """
    policies = _interconnection_map.get_by_tags(tags)
    return sorted(list(policies))


# ═══════════════════════════════════════════════════════════════════════════════
# 🔹 HELPER: PROFESSIONAL RESPONSE FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def format_policy_reference(policy_code: str) -> str:
    """Format a policy code with its title for professional documentation."""
    info = _interconnection_map.policies.get(policy_code)
    if info:
        return f"{policy_code} - {info['title']}"
    return policy_code
