#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# http https://search.rcsb.org/rcsbsearch/v2/query\?json\="{
#   \"query\": {
#     \"type\": \"terminal\",
#     \"label\": \"text\",
#     \"service\": \"text\",
#     \"parameters\": {
#       \"attribute\": \"entity_poly.rcsb_entity_polymer_type\",
#       \"operator\": \"exact_match\",
#       \"negation\": false,
#       \"value\": \"Protein\"
#     }
#   },
#   \"return_type\": \"polymer_entity\",
#   \"request_options\": {
#     \"paginate\": {
#       \"start\": 0,
#       \"rows\": 25
#     },
#     \"results_content_type\": [
#       \"experimental\"
#     ],
#     \"sort\": [
#       {
#         \"sort_by\": \"score\",
#         \"direction\": \"desc\"
#       }
#     ],
#     \"scoring_strategy\": \"combined\"
#   }
# }"

http https://search.rcsb.org/rcsbsearch/v2/query\?json\="{
  \"query\": {
    \"type\": \"terminal\",
    \"label\": \"text\",
    \"service\": \"text\",
    \"parameters\": {
      \"attribute\": \"entity_poly.rcsb_entity_polymer_type\",
      \"operator\": \"exact_match\",
      \"negation\": false,
      \"value\": \"Protein\"
    }
  },
  \"request_options\": {
    \"return_all_hits\": true
  },
  \"return_type\": \"polymer_entity\"
}" > protein_list.json
