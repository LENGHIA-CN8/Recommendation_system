import aerospike
from tqdm import tqdm
from datetime import datetime, timedelta


class Aerospike:
    def __init__(self, cluster, debug=False):
        hosts = [(str(h), p) for h, p in cluster]
        self.config = {"hosts": cluster}

        # Initial client
        self.client = aerospike.client({"hosts": cluster}).connect()
#         self.get_connection()

        if debug:
            try:
                self.client.connect()
                self.client.close()
            except Exception as e:
                self.context.logger.exception(e)
                
    def get_connection(self):
        if self.client is None:
            self.client = aerospike.client(self.config)
            self.client.connect()
        elif not self.client.is_connected():
            self.client.connect()
        return self.client
    
    def get(self, namespace, _set, _bin, _key):
        key = (namespace, _set, _key)
        (key, meta) = self.client.exists(key)
        data = None
        if meta != None:
            (key, meta, bins) = self.client.get(key)
            data = bins[_bin]
        return data
    
    def get_active_user_time(self, days = 3):
        # Get the current date
        current_date = datetime.now()

        # Calculate the date three days ago
        three_days_ago = current_date - timedelta(days=days)

        # Generate dates from today to three days ago
        list_user = set()
        dates_list = []
        while current_date >= three_days_ago:
            dates_list.append(current_date.strftime('%Y-%m-%d'))
            current_date -= timedelta(days=1)
            
        for d in dates_list:
            list_id = self.get('mem_storage', 'cafef_users_active_statistic', 'usersActive', d)
            list_user.update(list_id)
            
        return list(list_user)

if __name__ == "__main__":
    cluster = [("172.26.49.69", 3500), ("172.26.49.70", 3500), ("172.26.49.71", 3500), ("172.26.49.72", 3500)]
    print(cluster)
    aero_client = Aerospike(cluster, debug=False)
    list_user_ids = aero_client.get_active_user_time(3)
    print(list_user_ids[:10])

    
