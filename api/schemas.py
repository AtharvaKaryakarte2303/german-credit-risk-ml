from pydantic import BaseModel

class CreditData(BaseModel):
    duration: float
    credit_amount: float
    age: float
    checking_account: float
    savings_account: float
    employment_since: float
    installment_rate: float
    housing: float
    job: float
