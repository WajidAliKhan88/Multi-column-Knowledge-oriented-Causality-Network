import ujson as json


def load_json(path):
    with open(path, 'r') as fh:
        error = json.load(fh)
    return error


mckn_path = '../outputs/bootstrapped/MCKN/results/FALSE_valid.json'
tb_path = '../outputs/bootstrapped/TB/results/FALSE_valid.json'
mckn_error = load_json(mckn_path)
scrn_error = load_json(scrn_path)
tb_error = load_json(tb_path)

mckn_error = set(mcdn_error['FP'] + mckn_error['FN'])
scrn_ids = set(scrn_error['FP'] + scrn_error['FN'])
tb_ids = set(tb_error['FP'] + tb_error['FN'])

ms_com = mckn_ids.intersection(scrn_ids)
st_com = scrn_ids.intersection(tb_ids)
mst_dif = st_com - mckn_ids
tsm_dif = tb_ids - mckn_ids - scrn_ids
stm_dif = scrn_ids - tb_ids - mcdn_ids
ms_dif = scrn_ids - mckn_ids
sm_dif = mckn_ids - scrn_ids
mt_dif = tb_ids - mckn_ids
print(len(mst_dif), mst_dif)
print(len(tsm_dif), tsm_dif)
print(len(stm_dif), stm_dif)
# print(len(ms_com), ms_com)
# print(len(ms_dif), ms_dif)
# print(len(sm_dif), sm_dif)
# print(len(mt_dif), mt_dif)
print('hello world')
