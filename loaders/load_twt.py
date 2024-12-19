import glob
import math

import datetime as dt

import torch
from torch_geometric.data import Data
from tqdm import tqdm


DATA_DIR = '/mnt/raid1_ssd_4tb/datasets/twitter_misinfo/'

TWT_LEN = 31
TWEET_TXT = 12
USR_LOCATION = 3
USR_BIO = 4
USR_LEN = 10

MENTION = 0
REPLY = 1
RETWEET = 2

def get_twt_time(s):
    s = s.replace(',"', '')
    return dt.datetime.strptime(
        s, "%Y-%m-%d %H:%M"
    ).timestamp()

def has_line(file):
    cur_pos = file.tell()
    does_it = bool(file.readline())
    file.seek(cur_pos)
    return does_it

def tokenize(f, expected_tokens=None):
    # Trim off leading and tailing quote char (plus newline for tail)
    line = f.readline()

    # Annoyingly, some files use "," as the delimeter, and others use ,
    if line.startswith('"'):
        LEADING_CHAR = True
        DELIMITER = '","'
        TRAILING_CHAR = True
    else:
        LEADING_CHAR = False
        DELIMITER = ","
        TRAILING_CHAR = False

    if LEADING_CHAR:
        line = line[1:]

    # Sometimes tweet text has \n's in it so we can't fully rely on
    # readline to give us a correct full line read
    tokens = line.split(DELIMITER)

    if expected_tokens:
        while len(tokens) < expected_tokens:
            line += f.readline()
            tokens = line.split(DELIMITER)

    # So many problems when they don't use quotes around every token
    if DELIMITER == ',' and expected_tokens:
        for i in range(expected_tokens):
            t = tokens[i]

            # Generally, things with commas and newlines in them are quoted
            while t.startswith('"') and not t.endswith('"'):
                txt = tokens.pop(i+1)
                tokens[i] += DELIMITER + txt
                t = tokens[i]

                # Newlines in the bio can also break the parser
                if len(tokens) < expected_tokens:
                    new_line = f.readline()
                    line += new_line
                    tokens += new_line.split(DELIMITER)

    # Trim off trailing newline
    tokens[-1] = tokens[-1][:-1]

    # Trim off trailing quote (if necessary)
    if TRAILING_CHAR:
        tokens[-1] = tokens[-1][:-1]

    # Somehow, tweet text is still breaking this. Just assume that's the problem
    # and merge all tokens back into tweet text
    if expected_tokens:
        while len(tokens) > expected_tokens:
            if expected_tokens == TWT_LEN:
                txt = tokens.pop(TWEET_TXT+1)
                tokens[TWEET_TXT] += DELIMITER + txt
            else:
                print("Hm")

    return tokens

def jsonify(f, keys, expected_len):
    db = dict()
    tokens = tokenize(f, expected_len)
    for i,t in enumerate(tokens):
        db[keys[i]] = t
    return db

def build_keymap(first_line):
    tokens = tokenize(first_line)
    return tokens

def parse_tweet(f, keymap, usr_map):
    line = jsonify(f, keymap, TWT_LEN)

    usr = get_or_add(line['userid'], usr_map)
    twt_time = line['tweet_time']
    t = get_twt_time(twt_time)

    # Any of these could be edges
    mentions = line['user_mentions']

    try:
        mentions = eval(mentions) # Convert to list of str user ids

    # In some files, they're inconsistant about if no mentinos
    # should be represented as [] or ""
    except:
        mentions = []

    reply = line['in_reply_to_userid']
    rt = line['retweet_userid']

    src = []
    dst = []
    etype = []
    ts = []

    # Add any edges that were present in the tweet
    if mentions:
        src += [usr] * len(mentions)
        dst += [get_or_add(m, usr_map) for m in mentions]
        etype += [MENTION] * len(mentions)
        ts += [t] * len(mentions)

    if reply:
        src.append(usr)
        dst.append(get_or_add(reply, usr_map))
        etype.append(REPLY)
        ts.append(t)

    if rt:
        src.append(usr)
        dst.append(get_or_add(rt, usr_map))
        etype.append(RETWEET)
        ts.append(t)

    return src,dst,etype,ts

def get_or_add(key, db):
    if (uuid := db.get(key)) is None:
        uuid = len(db)
        db[key] = uuid

    return uuid

def parse_user(f, keymap, usr_map, lang_map, cur_year):
    line = jsonify(f, keymap, USR_LEN)
    id = line['userid']
    id = get_or_add(id, usr_map)

    # Do some rudimentary feature engineering here
    lang = get_or_add(line['account_language'], lang_map)
    followers = 0 if line['follower_count'] == '' else float(line['follower_count'])
    following = 0 if line['following_count'] == '' else float(line['following_count'])
    ratio = followers / (following+1)
    has_url = 1 if line['user_profile_url'] else 0
    created = dt.datetime.strptime(line['account_creation_date'], "%Y-%m-%d").year
    acct_age = cur_year - created

    # Stack all of the individual features together, and keep
    # language separate because it's one-hot and we don't know
    # how many langs there are until the end
    x = [
        math.log10(followers+1),
        math.log10(following+1),
        ratio,
        has_url,
        acct_age
    ]

    return id, x, lang

def parse_usr_file(fname, cur_year, rows=[], langs=[], usr_map=dict(), lang_map=dict()):
    f = open(fname, 'r')
    columns = build_keymap(f)
    prog = tqdm()

    while has_line(f):
        uuid, x, lang = parse_user(f, columns, usr_map, lang_map, cur_year)
        prog.update()

        if uuid >= len(rows):
            rows.append(x)
            langs.append(lang)

        # In the unlikely scenario that a row is repeated,
        # I guess replace the old features?
        else:
            print(f"Repeated user: {uuid}")
            rows[uuid] = x
            langs[uuid] = lang

    prog.close()
    return rows, langs, usr_map, lang_map

def parse_tweet_file(fname, usr_map):
    '''
    Reads tweet data file and converts any user interactions
    into edges between those users. For now, leave as strings
    bc we need to parse the user file to get uuids
    '''
    f = open(fname, 'r')
    columns = build_keymap(f)

    prog = tqdm()
    src,dst,etype,ts = [],[],[],[]
    while has_line(f):
        s,d,e,t = parse_tweet(f, columns, usr_map)
        src += s; dst += d; etype += e; ts += t
        prog.update()

    prog.close()

    data = Data(
        edge_index = torch.tensor([src,dst]),
        edge_attr  = torch.tensor(etype),
        ts = torch.tensor(ts)
    )

    return data

def parse_campaign(dir_name):
    # Files are named e.g. aug2019/filename.csv
    year = int(dir_name.split('/')[-2][-4:])

    usr_files = glob.glob(f'{DATA_DIR}/{dir_name}*_users_*.csv')
    twt_files = glob.glob(f'{DATA_DIR}/{dir_name}*_tweets_*.csv')

    print("Getting user data")

    # Parse each user file
    rows,langs,usr_map,lang_map = [],[],dict(),dict()
    for f in usr_files:
        rows,langs,usr_map,lang_map = parse_usr_file(
            f, year, rows, langs, usr_map, lang_map
        )

    # Build one-hot vector of user langauges
    one_hot = torch.zeros(len(rows), len(lang_map))
    for i,l in enumerate(langs):
        one_hot[i, l] = 1.

    # Concat with standard feature vector
    x = torch.cat([
        torch.tensor(rows),
        one_hot
    ], dim=1)

    print("Getting edge data")
    graphs = []
    for f in twt_files:
        g = parse_tweet_file(f, usr_map)
        graphs.append(g)

    graph = Data(
        edge_index = torch.cat([g.edge_index for g in graphs], dim=1),
        edge_attr = torch.cat([g.edge_attr for g in graphs]),
        ts = torch.cat([g.ts for g in graphs])
    )

    # Check if new nodes were added that weren't present in
    # the user data files. For now, just zero-pad the feature matrix
    if graph.edge_index.max() > x.size(0):
        diff = graph.edge_index.max()+1 - x.size(0)

        print(f"Adding features for {diff} unknown users")
        zeros = torch.zeros(diff, x.size(1))
        x = torch.cat([x, zeros])

    print("Done")
    graph.x = x
    return graph

if __name__ == '__main__':
    #g = parse_campaign('aug2019')
    #torch.save(g, '../graphs/aug2019.pt')

    to_parse = [
        #'jan2019/bangladesh', # 15 nodes, 4935 edges
        #'jan2019/iran',    # Eval
        #'jan2019/russia',   # Eval
        #'jan2019/venezuela',# Eval (crashed)
        #'june2019/catalonia',
        #'june2019/iran',
        #'june2019/russia',
        #'june2019/venezuela'
        'aug2019/china',    # Eval
        #'oct2018/', # All iran
        #'sept2019/china',
        #'sept2019/egypt',
        #'sept2019/saudi_arabia',
        #'sept2019/spain',
        'sept2019/uae'      # Eval 
    ]
    for f in to_parse:
        g = parse_campaign(f)
        print(f'{f}: {g.x.size(0)} nodes, {g.edge_index.size(1)} edges')
        torch.save(g, f'../graphs/{f.replace("/", "_")}.pt')