import json
import jwt
import time
import datetime

from hyper import HTTPConnection

def send_notification():
    ALGORITHM = 'ES256'

    APNS_KEY_ID = '27LKLT6JA6'
    APNS_AUTH_KEY = 'AuthKey_27LKLT6JA6.p8'
    TEAM_ID = 'N87XX7JNU3'
    BUNDLE_ID = 'edu.cmu.iw.chenli.comfort'
    #REGISTRATION_ID = '48e6081edca89c945ea15abe9669da1ebad165e2ac348bfb478e143154685012'
    REGISTRATION_ID = '1dba1ad3032432fc99db9c49a23a2ce206fed105aa40a8ae99b6480adad8557d'

    f = open(APNS_AUTH_KEY)
    secret = f.read()

    key_time = time.time() + 30
    print(
        datetime.datetime.fromtimestamp(
            key_time
        ).strftime('%Y-%m-%d %H:%M:%S')
    )

    token = jwt.encode(
        {
            'iss': TEAM_ID,
            'iat': key_time 
        },
        secret,
        algorithm= ALGORITHM,
        headers={
            'alg': ALGORITHM,
            'kid': APNS_KEY_ID,
        }
    )

    path = '/3/device/{0}'.format(REGISTRATION_ID)

    request_headers = {
        'apns-expiration': '0',
        'apns-priority': '10',
        'apns-topic': BUNDLE_ID,
        'authorization': 'bearer {0}'.format(token.decode('ascii'))
    }


    # Open a connection the APNS server
    conn = HTTPConnection('api.development.push.apple.com:443')

    payload_data = { 
        "aps": {
            "badge": 0,
            "content-available": 1,
        },
        "custom" : {
            "title": "Click on the screen:",
            'message': "Tell me your current thermal sensation!"
        }
    }
    payload = json.dumps(payload_data).encode('utf-8')

    # Send our request
    conn.request(
        'POST', 
        path, 
        payload, 
        headers=request_headers
    )
    resp = conn.get_response()
    print(resp.status)
    print(resp.read())
    return

while True:
    send_notification()
    time.sleep(10)

