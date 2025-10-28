from pydantic import BaseModel

class CreditData(BaseModel):
    duration_in_month: float
    credit_amount: float
    installment_rate: float
    age: float
    existing_credits: int
    num_dependents: int
    checking_account_status: str
    savings_account_status: str
    credit_history: str
    purpose: str
    employment: str
    personal_status: str
    other_debtors: str
    property: str
    other_installment_plans: str
    housing: str
    job: str
    telephone: str
    foreign_worker: str
