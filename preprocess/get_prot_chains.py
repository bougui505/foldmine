#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import json
import requests
import numpy as np
import urllib.parse


# http https://data.rcsb.org/graphql\?query\="{
#   polymer_entities(entity_ids: [
#     $ENTITIES
#     ])
#   {
#     rcsb_id
#     entry {
#       rcsb_entry_container_identifiers {
#         entry_id
#       }
#     }
#     polymer_entity_instances {
#       rcsb_polymer_entity_instance_container_identifiers {
#         auth_asym_id
#       }
#     }
#   }
# }"
def chunks(L, n):
    return [L[x:x + n] for x in range(0, len(L), n)]


nbatch = 100

jsonfile = open('protein_list.json', 'r')
data = json.loads(jsonfile.read())
idlist = [i['identifier'] for i in data['result_set']]
batches = chunks(idlist, nbatch)
# params = {
#     'query':
#     '{polymer_entities(entity_ids: ["101M_1","102L_1","102M_1","103L_1","103M_1"]) {rcsb_id entry {rcsb_entry_container_identifiers {entry_id}}    polymer_entity_instances {rcsb_polymer_entity_instance_container_identifiers {auth_asym_id}}}}'
# }
with open('protein_chains.txt', 'w') as outfile:
    for batch in batches:
        query = '{polymer_entities(entity_ids: ["%s"]) {rcsb_id entry {rcsb_entry_container_identifiers {entry_id}}    polymer_entity_instances {rcsb_polymer_entity_instance_container_identifiers {auth_asym_id}}}}' % '","'.join(
            batch)
        url = f'https://data.rcsb.org/graphql?query={query}'
        r = requests.get(url)
        for e in r.json()['data']['polymer_entities']:
            pdb = e['entry']['rcsb_entry_container_identifiers']['entry_id']
            chain = e['polymer_entity_instances'][0]['rcsb_polymer_entity_instance_container_identifiers'][
                'auth_asym_id']
            outfile.write(f'{pdb}_{chain}\n')
