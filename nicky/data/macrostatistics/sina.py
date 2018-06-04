import ast
import json
import urllib.request

import pandas as pd


class sina(object):
    __base_api = 'http://money.finance.sina.com.cn/mac/api/jsonp.php/SINAREMOTECALLCALLBACK/MacPage_Service.get_pagedata?cate={}&event={}&from=0&num=31&condition='
    __cate_info = {
    "nation": 14,
    "price": 13,
    "resident": 3,
    "fixed": 11,
    "boom": 6,
    "foreign": 6,
    "fininfo": 24,
    "finance": 20,
    "industry": 22,
    }

    def get_hist_data(cate):
        if cate not in sina.__cate_info:
            return None
        data = {}
        for event in range(sina.__cate_info[cate]):
#            try:
            if 1:
                data[event] = sina._get_mkt_hist_from_sina(cate, event)
#            except:
                print(cate, event)

    def _get_mkt_hist_from_sina(cate, event):
        url = sina.__base_api.format(cate, event)
        data = urllib.request.urlopen(urllib.request.Request(
                url,
                headers={
                    "Accept-Encoding": "deflate",
                    "User-Agent": "Mozilla/5.0 (X11; U; Linux i686) Gecko/20071127 Firefox/2.0.0.11",
                })).read().decode('gbk')
        data = sina._handle_dirty_json_data(data)
        data = sina._json_to_pandas(data)
        return data

    def _json_to_pandas(data):
        columns = data.get('config').get('all')
        columns = [column[1] for column in columns]
        records = data.get('data')
        data = pd.DataFrame(records, columns=columns)
        return data

    def _handle_dirty_json_data(data):
        lstrip = "//<script>location.href='http://sina.com.cn'; </script>\nSINAREMOTECALLCALLBACK(("
        rstrip = "))"
        data = data.lstrip(lstrip).rstrip(rstrip)
        data = data.replace(',""', '')
        data = data.replace("'", '"')
        data = sina._fix_json(data)
        try:
            data = json.loads(data)
        except:
            print(data)
        return data


    def all():
        for cate in sina.__cate_info:
            sina.get_hist_data(cate)

    def _fix_json(string):
        x = []
        for i, s in enumerate(string):
            x.append(s)
            if s == ":" and (string[i-1].islower() or string[i-1].isupper()):
                j = i - 1
                while string[j].islower() or string[j].isupper():
                    x.pop()
                    j -= 1
                x.pop()
                x.append('"')
                x.extend(list(string[j+1:i]))
                x.append('":')
        x = ''.join(x)
        return x

