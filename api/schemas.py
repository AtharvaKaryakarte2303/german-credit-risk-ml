from pydantic import BaseModel

class CreditData(BaseModel):
    duration: float
    credit_amount: float
    age: float
    checking_status_A11: int
    checking_status_A12: int
    checking_status_A13: int
    savings_status_A61: int
    savings_status_A62: int
    savings_status_A63: int
    employment_A71: int
    employment_A72: int
    employment_A73: int
