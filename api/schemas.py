from pydantic import BaseModel

class CreditData(BaseModel):
    checking_account_status: str
    duration_in_month: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account_status: str
    employment: str
    installment_rate: int
    personal_status: str
    other_debtors: str
    present_residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    telephone: str
    foreign_worker: str
