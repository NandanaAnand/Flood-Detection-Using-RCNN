"""
Description:Stores Twilio credentials and sends SMS alerts using the Twilio API.
            Used by other scripts for sending notifications when flooding thresholds are crossed.
"""

import twilio_sms_config

account_sid = 'insert your account sid'
auth_token = 'insert your auth_token'
twilio_number = 'insert your twilio number'
target_number = 'insert your target number'

from twilio.rest import Client
import twilio_sms_config
client= Client(keys_sms.account_sid, keys_sms.auth_token)
message = client.messages.create(

    body="Flood Imminent!",
    from_=keys_sms.twilio_number,
    to=keys_sms.target_number

)
print(message. body)