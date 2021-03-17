from urllib.parse import quote
from urllib.request import urlopen
from requests_oauthlib import OAuth1
import requests
import numpy as np
import math
from requests_oauthlib import OAuth1
import requests
import simplejson as json
from ediblepickle import checkpoint
import os
import time
import collections
import pickle
import heapq
import traceback

# users must have at least this many followers or tweets
follower_cut_off = 25
tweet_cut_off = 100

BETA = 2 / np.log(2)  # = GAMMA_N/ln(2) determines shape of gamma values
rng = np.random.default_rng()

with open("twitter_secrets.json.nogit") as fh:
    secrets = json.loads(fh.read())

WAIT_TIME = 15 * 60  # time in seconds needed to wait between 429 codes

# create an auth object
auth = OAuth1(
    secrets["api_key"],
    secrets["api_secret"],
    secrets["access_token"],
    secrets["access_token_secret"]
)

cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

FULL_DATA = []


# finds the size divided by the boundary count of a data point
def ratio(dat):
    return dat[1] / dat[2]


def twitter_percolation(screen_name, N_stop=1000, N_min=100, NA=100, tol=1e-6, save_file=None, watchQ=False):
    global FULL_DATA
    if save_file is None:
        save_file = f'{screen_name}_network.txt'

    user0_id = int(get_id(screen_name))

    network = SocialNetwork(user0_id, p_cut0=0.1)

    if os.path.exists(save_file):
        with open(save_file, 'rb') as fp:
            FULL_DATA = pickle.load(fp)

    for i in range(NA):
        network = SocialNetwork(user0_id, p_cut0=network.get_p_cut())
        FULL_DATA.append(network.data_point())
        size = 0
        change = False
        while network.size() < N_stop or not change:
            dat_prev = network.data_point()
            try:
                change = network.next_p()
            except IndexError:
                if watchQ:
                    print("Hit Dead End")
                break
            '''
            if network.p - FULL_DATA[-1][0] > 0:
                rate_change = (network.size() - FULL_DATA[-1][1]) / (network.p - FULL_DATA[-1][0])
                if watchQ and rate_change > max_rate:
                    print(f'rate = {rate_change}, size = {network.size()}, p = {network.p}')
                    max_rate = rate_change
                if network.size() > N_min and rate_change > deriv:
                    FULL_DATA.append(dat_prev)
                    break
            '''
            if watchQ and size < network.size():
                size = network.size()
                print(f'size = {network.size()}, p={network.p}')

            if change:
                FULL_DATA.append(dat_prev)

        FULL_DATA.append((network.p, network.size()))
        with open(save_file, 'wb') as fp:
            if watchQ:
                print('Saving')
            pickle.dump(FULL_DATA, fp)

    return FULL_DATA


@checkpoint(key=lambda args, kwargs: f'get_id_{args[0]}.p', work_dir=cache_dir)
def get_id(screen_name):
    user_id = requests.get(
        'https://api.twitter.com/2/users/by/username/' + screen_name,
        auth=auth
    )
    while user_id.status_code == 429:
        time.sleep(60)
        user_id = requests.get(
            'https://api.twitter.com/2/users/by/username/' + screen_name,
            auth=auth
        )
    return user_id.json()['data']['id']


SUCCESS_TIME = time.perf_counter()


@checkpoint(key=lambda args, kwargs: f'get_followers_{args[0]}_{args[1]}.p', work_dir=cache_dir)
def get_followers_by_page(user_id, pagination):
    global SUCCESS_TIME
    followers = requests.get(
        "https://api.twitter.com/2/users/" + str(user_id) + '/followers',
        auth=auth,
        params={'pagination_token': pagination, 'max_results': 1000, 'user.fields': 'public_metrics'}
    )
    request_time = time.perf_counter()
    factor = 4
    if followers.status_code == 429 or followers.status_code == 503:
        print(f'{user_id} -waiting... ', end='')
        if followers.status_code == 503:
            print(f'code: {followers.status_code} ', end='')
            factor *= 2
    elif followers.status_code != 200:
        raise Exception(f'Other error code: {followers.status_code}')

    i = 0
    t_wait = -time.perf_counter()
    while followers.status_code == 429 or followers.status_code == 503:
        '''
        dt = 450 / 2 ** i
        if dt < 10 ** -10:
            dt = 10 ** -10
        time.sleep(dt)
        i += 1
        '''
        i += 1
        sleep_time = WAIT_TIME/factor - time.perf_counter() + SUCCESS_TIME
        # print(f'sleeping for {sleep_time/60} ', end='')
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            time.sleep(1)

        followers = requests.get(
            "https://api.twitter.com/2/users/" + str(user_id) + '/followers',
            auth=auth,
            params={'pagination_token': pagination, 'max_results': 1000, 'user.fields': 'public_metrics'}
        )

    SUCCESS_TIME = time.perf_counter()

    if i != 0:
        print(f'waited time = {(t_wait + time.perf_counter()) / 60} mins')

    try:
        followers = followers.json()
    except Exception as e:
        # traceback.print_exc()
        print(f"Bad json - Got: {followers}")
        raise e

    if 'errors' in followers or 'data' not in followers:  # and followers['errors'] == 'Authorization Error':
        # print("ERROR:", followers)
        return [], {}

    # reduces saved information
    if 'next_token' in followers['meta']:
        meta = {'next_token': followers['meta']['next_token']}
    else:
        meta = {}

    followers = [{'id': f['id'],
                  'public_metrics': {'followers_count': f['public_metrics']['followers_count'],
                                     'tweet_count': f['public_metrics']['tweet_count']
                                     }
                  } for f in followers['data']]

    return followers, meta


def find_followers(user):
    user_id = user.id
    followers, next_page = get_followers_by_page(user_id, None)

    for f in followers:
        yield Follows(User(int(f['id'])), user)

    while 'next_token' in next_page:
        next_page = next_page['next_token']
        followers, next_page = get_followers_by_page(user_id, next_page)

        for f in followers:
            if f['public_metrics']['followers_count'] > follower_cut_off and f['public_metrics'][
                'tweet_count'] > tweet_cut_off:
                yield Follows(User(int(f['id'])), user)


# adds edge/node to components in sorted manner
# adds between start and end where end is not included
# always positive start and end
# returns added/found position
# doesn't add duplicates and returns -pos-1 if duplicate is found
def insert_into_list(sorted_list, item, start=0, end=None):
    if end is None:
        end = len(sorted_list)

    mid = (start + end) // 2

    if start == end:  # length of list is 0
        sorted_list.append(item)
        return 0
    elif end == start + 1:  # only 1 item left to compare and insert
        if sorted_list[start] == item:  # item already exists
            return -start - 1
        elif sorted_list[start] < item:
            sorted_list.insert(start, item)
            return start
        else:
            sorted_list.insert(start + 1, item)
            return start + 1
    elif sorted_list[mid] < item:  # item must be added strictly below mid
        return insert_into_list(sorted_list, item, start=start, end=mid)
    else:  # item must be added above mid inclusively
        return insert_into_list(sorted_list, item, start=mid, end=end)


class SortedList:
    def __init__(self, initial=[]):
        self.list = initial
        heapq.heapify(self.list)

    def __str__(self):
        return f'{self.list}'

    # inserts item into list
    # if item already exists then does nothing
    def push(self, item):
        # pos = insert_into_list(self.list, item)
        heapq.heappush(self.list, item)

    def pop(self):
        # return self.list.pop()
        return heapq.heappop(self.list)

    def peak(self):
        # return self.list[-1]
        return self.list[0]

    # takes in a new list to push onto new_list,
    # if list is already sorted set sortQ = False
    def merge(self, new_list, sortQ=True):
        if len(new_list) == 0:
            return None

        if sortQ:
            # new_list = sorted(new_list, reverse=True)
            heapq.heapify(new_list)

        self.list = [el for el in heapq.merge(self.list, new_list)]
        '''
        index = 0
        i = 0
        for item in new_list:
            while index < len(self.list) and item < self.list[index]:
                index += 1

            if index >= len(self.list):
                break
            elif item == self.list[index]:
                continue
            else:
                self.list.insert(index, item)
            i += 1

        if i < len(new_list):
            self.list.extend(new_list[i:])
        '''

    def size(self):
        return len(self.list)


# stores a large sorted list of components
# small p_values are put on a heap, while large values are put in an unordered stack
class SortedComponents:
    def __init__(self, p_cut=0.5, initial=[], dp=0.002):
        self.p_cut = p_cut
        self.dp = dp

        self.sorted = SortedList()  # sorted list of follows
        self.list = []  # unsorted list of follows
        self.user_list = []  # unsorted list of users

        for c in initial:
            if type(c) == USER_TYPE:
                self.user_list.append(c)
            elif c.p_value() <= p_cut:
                self.sorted.push(c)
            else:
                self.list.append(c)

        self.next_user = None
        self.get_next_user()

    def __str__(self):
        return f'Sorted Components: ' \
               f'(p_cut: {self.p_cut}, sorted: {self.sorted.size()}, ' \
               f'list: {len(self.list)}, user: {len(self.user_list)})'

    def size(self):
        return self.sorted.size() + len(self.list) + len(self.user_list)

    def users_size(self):
        return len(self.user_list)

    # returns next user and updates self.next_user
    def get_next_user(self):
        tmp = self.next_user
        try:
            self.next_user = min(self.user_list)
            self.user_list.remove(self.next_user)
        except ValueError:
            self.next_user = None
        return tmp

    def pop(self):
        try:
            if self.next_user is None:
                return self.sorted.pop()

            next_follow = self.sorted.peak()
            if self.next_user < next_follow:
                return self.get_next_user()
            elif self.next_user.p_value() == next_follow.p_value() and rng.random() < 0.5:
                return self.sorted.pop()
            elif self.next_user.p_value() == next_follow.p_value():
                return self.get_next_user()
            else:
                return self.sorted.pop()

        except IndexError:
            if len(self.list) == 0:
                if self.next_user is None:
                    return None
                else:
                    return self.get_next_user()

            self.p_cut += self.dp
            if self.p_cut > 1:
                self.p_cut = 1
            self.update_sorted()
            return self.pop()

    def push(self, item):
        if type(item) == USER_TYPE:
            if self.next_user is None:
                self.next_user = item
                return None

            try:
                index = self.user_list.index(item)
            except ValueError:
                index = None

            if item < self.next_user:
                self.user_list.append(self.next_user)
                self.next_user = item
                if index is not None:
                    del self.user_list[index]
            else:
                if item is None:
                    self.user_list.append(item)
        elif item.p_value() <= self.p_cut:
            self.sorted.push(item)
        else:
            self.list.append(item)

    # moves elements from self.list into self.sorted
    def update_sorted(self):
        delete = []
        for i, c in enumerate(self.list):
            if c.p_value() <= self.p_cut:
                self.sorted.push(c)
                delete.append(i)
        delete.sort(reverse=True)
        for i in delete:
            del self.list[i]

    # takes in another SortedComponents class and merges it into this class
    # heap/self will change their p_cut to larger value
    def merge(self, heap):
        if heap.p_cut != self.p_cut:
            if heap.p_cut > self.p_cut:
                self.p_cut = heap.p_cut
                self.update_sorted()
            else:
                heap.p_cut = self.p_cut
                self.update_sorted()

        self.list.extend(heap.list)
        self.user_list.extend(heap.user_list)
        self.sorted.merge(heap.sorted.list)

    def merge_list(self, new_list):
        # self.merge(SortedComponents(p_cut=self.p_cut, dp=self.dp, initial=new_list))
        for item in new_list:
            self.push(item)


# class to be inherited which contains relations for id and p_value
class Component:
    def __init__(self):  # should be over written
        self.p_val = None
        self.id = None

    def p_value(self):  # should be over written
        return self.p_val

    # definition of comparators
    # equal use id while comparators use p_value
    def __lt__(self, other):
        return self.p_value() < other.p_value()

    def __le__(self, other):
        return self.p_value() <= other.p_value()

    def __gt__(self, other):
        return self.p_value() > other.p_value()

    def __ge__(self, other):
        return self.p_value() >= other.p_value()

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id


# edge class where user follower follows influencer
# saves p-value which is assigned randomly at creating
class Follows(Component):
    def __init__(self, follower, influencer):
        self.follower = follower
        self.influencer = influencer
        self.p_val = np.random.random()
        self.id = -influencer.id - follower.id - 1  # exists to allow equal comparison between components

    def __str__(self):
        return 'Edge: {0} follows {1} with p={2}'.format(self.follower.id, self.influencer.id, self.p_val)

    def p_value(self):
        return self.p_val


# class which represents users
# stores their p_m, gamma values followers and number of open influencer
class User(Component):
    def __init__(self, ID, beta=BETA):
        self.id = ID
        # self.followers = [Follows(User(i), self) for i in find_followers(ID)]  # saves followers as edges
        self.n_inf = 0
        self.p_m = 0.211544 * rng.random() * 2
        if beta == 0:
            self.gamma = None
            self.p_m = rng.random()
        else:
            self.gamma = rng.exponential(scale=beta)
        self.update = False
        self.p_val = 1

        # sorted(self.followers, key=lambda x: x.p_value()) # keeps followers sorted

    def __str__(self):
        return '{0}: (influences {1}, pm {2}, gamma {3}, p_value {4})'.format(self.id, self.n_inf, self.p_m, self.gamma,
                                                                              self.p_value())

    def add_influencer(self):  # increases influcener count and keeps track to update p_value
        self.n_inf += 1
        self.update = True

    # returns calculated p_value
    def p_value(self):
        if self.update:
            if self.gamma is None:
                self.p_val = self.p_m
            else:
                self.p_val = (1 - self.p_m) * np.exp(-self.n_inf / self.gamma) + self.p_m
        return self.p_val


COMPONENT_TYPE = type(Component())
FOLLOWS_TYPE = type(Follows(User(0), User(1)))
USER_TYPE = type(User(0))


# Network class which holds found followers and open edges/nodes
# initialized with a single user id
class SocialNetwork:
    def __init__(self, ID, p_cut0=0.5, dp=0.002):
        self.open_users = set()  # set of open nodes, prevents adding closed node which is already open

        # ordered list of closed components
        self.components = SortedComponents(p_cut=p_cut0, dp=dp)

        self.p = 0  # current largest open p_value
        # opens given follower and adds their followers
        self.open_user(User(ID))
        self.p = 0  # forces starting p to equal 0

    def __str__(self):
        return f'Network: size = {self.size()}, p = {self.p}'

    def get_p_cut(self):
        return self.components.p_cut

    def next_p(self):  # opens the next smallest component, returns if p has changed
        next_comp = self.components.pop()

        p_prior = self.p
        if type(next_comp) is FOLLOWS_TYPE:
            self.open_follow(next_comp)
        elif type(next_comp) is USER_TYPE:
            self.open_user(next_comp)
        else:  # this should not happen give error
            raise ValueError(f'components contains non User or Follows type = {next_comp}')

        if p_prior != self.p:
            return True
        else:
            return False

    def open_follow(self, follow):  # opens the given follow then adds the follower
        if follow.p_value() > self.p:  # updates p if needed
            self.p = follow.p_value()

        # follow.follower needs to be either added or updated
        if follow.follower.id not in self.open_users:
            follow.follower.add_influencer()  # increases number of influencers
            self.add_user(follow.follower)  # adds their follower to components

    def open_user(self, user):  # opens the given user then finds their followers
        if user.p_value() > self.p:  # updates p if needed
            self.p = user.p_value()

        self.open_users.add(user.id)  # adds user to open list

        # now add all of user's followers
        for f in find_followers(user):
            if f.follower.id not in self.open_users:
                if f.p_value() < self.p:
                    self.open_follow(f)
                else:
                    self.components.push(f)

    # if user is new then adds user to component list
    # otherwise updates user's position
    def add_user(self, user):
        self.components.push(user)
        # pos = add_component(self.components, user)
        # self.update_component(pos)

    def add_follow(self, follow):  # adds a follow to component list
        # ignore follows to already open users
        if follow.follower.id not in self.open_users:
            # add_component(self.components, follow)
            self.components.push(follow)

    def size(self):  # returns number of open users
        return len(self.open_users)

    def boundary_size(self):  # returns the number of non opened users
        return self.components.users_size()

    def data_point(self):
        return (self.p, self.size(), self.size() + self.boundary_size())
