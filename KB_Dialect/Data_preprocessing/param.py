import re

# 발화 속 등장하는 이름, 주소, 고유명사 등을 전부 마스킹(다른 토큰으로 대체)
NAMES = ['NAME', 'NAEM', 'anem', 'anme', 'mane', 'naem', 'nam', 'nmae', '이름', '고자영', '최미영']
PAT_LIST = [
    '&NAEM4&', '&NAME&', '&NAME18&', '&adderess2&', '&adderss11&', '&address&', '&address1&',
    '&address10&', '&address11&', '&address12&', '&address13&', '&address14&', '&address15&',
    '&address16&', '&address17&', '&address18&', '&address19&', '&address2&', '&address20&',
    '&address21&', '&address22&', '&address23&', '&address3&', '&address4&', '&address5&',
    '&address6&', '&address7&', '&address8&', '&address9&', '&addressa&', '&adress&', '&anem6&',
    '&anme1&', '&anme5&', '&anme6&', '&company2&', '&company3&', '&company_name1&', '&company_name2&',
    '&mane1&', '&mane4&', '&mane5&', '&naem1&', '&naem16&', '&naem2&', '&naem6&', '&naem7&',
    '&naem9&', '&nam13&', '&nam16e&', '&nam1e&', '&nam3&', '&nam4&', '&nam51&', '&nam7&',
    '&namE5&', '&name&', '&name0&', '&name1&', '&name10&', '&name11&', '&name12&', '&name13&',
    '&name14&', '&name145&', '&name15&', '&name16&', '&name17&', '&name18&', '&name19&',
    '&name2&', '&name20&', '&name21&', '&name22&', '&name23&', '&name24&', '&name25&',
    '&name26&', '&name27&', '&name28&', '&name29&', '&name3&', '&name30&', '&name31&',
    '&name32&', '&name33&', '&name34&', '&name35&', '&name36&', '&name37&', '&name38&',
    '&name39&', '&name4&', '&name40&', '&name41&', '&name42&', '&name43&', '&name44&',
    '&name45&', '&name46&', '&name47&', '&name48&', '&name49&', '&name5&', '&name50&',
    '&name51&', '&name52&', '&name54&', '&name55&', '&name56&', '&name57&', '&name59&',
    '&name6&', '&name60&', '&name61&', '&name62&', '&name63&', '&name64&', '&name65&',
    '&name67&', '&name68&', '&name7&', '&name8&', '&name9&', '&names5&', '&nmae2&', '&nmae3&',
    '&가자&', '&고자영2&', '&상호명1&', '&상호명2&', '&서연림1&', '&선옥언니&', '&월령&',
    '&유튜브&', '&이름1&', '&이름2&', '&이름4&', '&이름5&', '&인가&', '&좌미영2&', '&한림농협&'
]

# 발화 속 등장하는 이름, 주소, 고유명사 등을 전부 마스킹(다른 토큰으로 대체)
def pat_clean():
    PAT_MAP = {}
    for p in PAT_LIST:
        PAT_MAP[p] = "[OTHER]"
        if "add" in p:
            PAT_MAP[p] = "[ADDRESS]"
        for n in NAMES:
            if n in p:
                PAT_MAP[p] = "[NAME]"
    return PAT_MAP

PAT_MAP = pat_clean()

# 정규표현식 패턴 (데이터셋에 이런 정규패턴이 등장하긴 함.)     
PATTERN1 = re.compile("(\(\(\)\))|(\{\w+\})|[#-]")
PATTERN2 = re.compile("(\(\(\w+\)\))")
PATTERN3 = re.compile("(&\w+&)")
PATTERN4 = re.compile("\((\w+)\)/\((\w+)\)")

