# -*- coding: utf-8 -*-
import servicelNotification
#task = '{"image":"https://scontent.fcgh26-1.fna.fbcdn.net/v/t1.0-9/1554539_10209192373635660_3166008603061972867_n.jpg?_nc_cat=102&_nc_eui2=AeH67ikIZ0pTBuMzqQcBAMLsYnK7ZvuLSbwArzPKr55WnEX-IL-q1tf-TMZtSFWoPIBpQKOhsHMH_clw5byEeLZCRZupKnKheYdgIldLdRbKnQ&_nc_ht=scontent.fcgh26-1.fna&oh=cf07621d1c3e660541ce012012025537&oe=5D6FD342","name":"Sumit kumar tiwari","message":"High end customer is entering"}'
notification = '{"name":"Sumit Kumar Tiwari", "message": "IMP Valuable Customer", "image":"https://www.aljazeera.com/mritems/imagecache/mbdxxlarge/mritems/Images/2019/5/17/fe9553a60cd74aef9a882c79c88cb45e_18.jpg"}'
### USING AN IMPORTED MODULE
# Use the form modulename.itemname
service = servicelNotification.NotificationCall()
service.callNotification(notification);

#Call Automated API Calling
toCall = '{"to":"+5511989067998"}'
#service.automatedCall(toCall)