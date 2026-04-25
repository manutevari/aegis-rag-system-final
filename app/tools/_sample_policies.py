"""Built-in sample policy corpus used when no POLICY_DIR is configured."""
from langchain.schema import Document

def get_sample_docs():
    policies = [
        ("T-04", "travel", """TRAVEL POLICY — T-04 (Effective 01-April-2025)

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

Source: Policy T-04"""),

        ("F-12", "reimbursement", """EXPENSE REIMBURSEMENT POLICY — F-12 (Effective 01-Jan-2025)

1. SUBMISSION DEADLINES
   0–30 days   : Submit via portal — standard processing
   31–60 days  : Requires direct manager approval
   61–90 days  : Requires VP approval
   > 90 days   : NOT reimbursed (no exceptions)

2. APPROVAL MATRIX
   Up to ₹50,000           : Direct Manager
   ₹50,001 – ₹2,00,000    : Manager + Finance BP
   Above ₹2,00,000         : VP + CFO

3. PAYMENT TIMELINES
   Approved claims paid within 10 working days by bank transfer.
   Advance settlements cleared within 5 days of trip completion.

4. NON-REIMBURSABLE ITEMS
   Alcohol, personal entertainment, traffic fines, upgrades beyond policy grade.

Source: Policy F-12"""),

        ("HR-07", "leave", """LEAVE POLICY — HR-07 (Effective 01-April-2025)

1. LEAVE ENTITLEMENTS (per year)
   Earned Leave (EL)    : 18 days (accumulates up to 90 days)
   Sick Leave (SL)      : 12 days (non-accumulative)
   Casual Leave (CL)    : 6 days (non-accumulative)
   Maternity Leave      : 26 weeks
   Paternity Leave      : 5 days
   Bereavement (immediate family) : 3 days
   Bereavement (extended family)  : 1 day

2. EL ENCASHMENT
   Allowed once per year — up to 15 days.
   Rate = Basic Salary / 26 × number of days.

3. NOTICE PERIODS
   1–3 days leave   : 2 working days notice
   4–10 days leave  : 5 working days notice
   > 10 days leave  : 15 working days + Manager approval

Source: Policy HR-07"""),

        ("C-03", "compensation", """COMPENSATION STRUCTURE — C-03 (Effective 01-April-2025)

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

Source: Policy C-03"""),

        ("IT-09", "IT assets", """IT ASSET POLICY — IT-09 (Effective 01-July-2024)

1. LAPTOP ENTITLEMENT
   L1-L3 : ₹55,000 budget | 4-year refresh
   L4-L5 : ₹85,000 budget | 3-year refresh
   L6-L7 : ₹1,20,000 budget | 3-year refresh
   VP+   : ₹1,80,000 budget | Premium + docking station

2. MOBILE DEVICE
   L5 and above : Corporate mobile, reimbursement up to ₹1,500/month
   Below L5     : BYOD, no reimbursement

3. WFH ALLOWANCE
   One-time setup  : ₹10,000 (all permanent employees on approved WFH)
   Monthly internet: Up to ₹1,000/month with receipt

4. LOSS / DAMAGE
   Accidental (1st) : Covered by company
   Accidental (2nd) : 50% employee liability
   Negligence       : 100% employee liability (depreciated value)

Source: Policy IT-09"""),
    ]
    return [Document(page_content=content, metadata={"policy_code": code, "category": cat})
            for code, cat, content in policies]
