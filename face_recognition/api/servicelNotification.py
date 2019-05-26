# -*- coding: utf-8 -*-
import requests
import json
class NotificationCall:
    @staticmethod
    def callNotification(messageObj):
        data = json.loads(messageObj)
        resp = requests.post('http://localhost:3000/pushmessages', json=data)
        if resp.status_code != 200:
            print ('Error: Server returned status code %s' % resp.get('status'))
            raise Exception
        print(resp)
        
    @staticmethod
    def automatedCall(messageObj):
        data = json.loads(messageObj)
        resp = requests.post('http://localhost:2000/automatedCall', json=data)
        if resp.status_code != 200:
            print ('Error: Server returned status code %s' % resp.get('status'))
            raise Exception
        print(resp)
