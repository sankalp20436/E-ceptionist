import yagmail
import os
import datetime

date = datetime.date.today().strftime("%B %d, %Y")
sub = "Request for meet " + str(date)
body = [
    "respected sir ",
    "a person wants to meet you",

    "wants to meet you for",

    "regards"
    "receptionist"
]

# mail information
yag = yagmail.SMTP("smarteceptionist@gmail.com", "minor@2022")

# sent the mail
yag.send(
    to="sankalpvarmadlw@gmail.com",
    subject=sub,  # email subject
    contents=body,  # email body

)
print("Email Sent!")
