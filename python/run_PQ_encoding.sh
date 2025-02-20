#/bin/bash

# python3 PQfirstimplementation_Kmeans.py

# python3 queryfirstimplementation.py --k 100 --db_file Kmeans_pq_index.db --codebook_file Kmeans_codebook.pkl > Kmeans_pq_index_100.txt
# python3 queryfirstimplementation.py --k 250 --db_file Kmeans_pq_index.db --codebook_file Kmeans_codebook.pkl > Kmeans_pq_index_250.txt
# python3 queryfirstimplementation.py --k 500 --db_file Kmeans_pq_index.db --codebook_file Kmeans_codebook.pkl > Kmeans_pq_index_500.txt

# python3 PQfirstimplementation_Kmedoids.py

# python3 queryfirstimplementation.py --k 100 --db_file Kmedoids_pq_index.db --codebook_file Kmedoids_codebook.pkl > Kmedoids_pq_index_100.txt
python3 queryfirstimplementation.py --k 250 --db_file Kmedoids_pq_index.db --codebook_file Kmedoids_codebook.pkl > Kmedoids_pq_index_250.txt
python3 queryfirstimplementation.py --k 500 --db_file Kmedoids_pq_index.db --codebook_file Kmedoids_codebook.pkl > Kmedoids_pq_index_500.txt

python3 PQfirstimplementation_DBscan.py

python3 queryfirstimplementation.py --k 100 --db_file DBScan_pq_index.db --codebook_file DBscan_codebook.pkl > DBScan_pq_index_100.txt
python3 queryfirstimplementation.py --k 250 --db_file DBScan_pq_index.db --codebook_file DBscan_codebook.pkl > DBScan_pq_index_250.txt
python3 queryfirstimplementation.py --k 500 --db_file DBScan_pq_index.db --codebook_file DBscan_codebook.pkl > DBScan_pq_index_500.txt

