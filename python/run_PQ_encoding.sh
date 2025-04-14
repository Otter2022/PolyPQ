#!/bin/bash

# python PQfirstimplementation_kmeans.py --m 147 --num_clusters 256 --codebook_out my_kmeans_codebook_k_256_m_147.pkl --pq_index_out my_kmeans_index_k_256_m_147.db

# python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmeans_index_k_256_m_147.db --codebook_file my_kmeans_codebook_k_256_m_147.pkl > kmeans_pq_index_100_k_256_m_147.txt

# # ------------------------
# # Set 1: m = 147, num_clusters = 512
# # ------------------------
# python PQfirstImplementation_kmeans.py --m 147 --num_clusters 512 --codebook_out my_kmeans_codebook_k_512_m_147.pkl --pq_index_out my_kmeans_index_k_512_m_147.db

# python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmeans_index_k_512_m_147.db --codebook_file my_kmeans_codebook_k_512_m_147.pkl > kmeans_pq_index_100_k_512_m_147.txt

# # ------------------------
# # Set 2: m = 147, num_clusters = 1024
# # ------------------------
# python PQfirstImplementation_kmeans.py --m 147 --num_clusters 1024 --codebook_out my_kmeans_codebook_k_1024_m_147.pkl --pq_index_out my_kmeans_index_k_1024_m_147.db

# python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmeans_index_k_1024_m_147.db --codebook_file my_kmeans_codebook_k_1024_m_147.pkl > kmeans_pq_index_100_k_1024_m_147.txt

# ------------------------
# Set 3: m = 49, num_clusters = 256
# ------------------------
python PQfirstImplementation_kmeans.py --m 49 --num_clusters 256 --codebook_out my_kmeans_codebook_k_256_m_49.pkl --pq_index_out my_kmeans_index_k_256_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmeans_index_k_256_m_49.db --codebook_file my_kmeans_codebook_k_256_m_49.pkl > kmeans_pq_index_100_k_256_m_49.txt

# ------------------------
# Set 4: m = 49, num_clusters = 512
# ------------------------
python PQfirstImplementation_kmeans.py --m 49 --num_clusters 512 --codebook_out my_kmeans_codebook_k_512_m_49.pkl --pq_index_out my_kmeans_index_k_512_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmeans_index_k_512_m_49.db --codebook_file my_kmeans_codebook_k_512_m_49.pkl > kmeans_pq_index_100_k_512_m_49.txt

# ------------------------
# Set 5: m = 49, num_clusters = 1024
# ------------------------
python PQfirstImplementation_kmeans.py --m 49 --num_clusters 1024 --codebook_out my_kmeans_codebook_k_1024_m_49.pkl --pq_index_out my_kmeans_index_k_1024_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmeans_index_k_1024_m_49.db --codebook_file my_kmeans_codebook_k_1024_m_49.pkl > kmeans_pq_index_100_k_1024_m_49.txt

# ------------------------

python PQfirstImplementation_Kmedoids.py --m 147 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_147.pkl --pq_index_out my_kmedoids_index_k_256_m_147.db

python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmedoids_index_k_256_m_147.db --codebook_file my_kmedoids_codebook_k_256_m_147.pkl > Kmedoids_pq_index_100_k_256_m_147.txt

# ------------------------
# Set 4: m = 1, num_clusters = 512
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 147 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_147.pkl --pq_index_out my_kmedoids_index_k_512_m_147.db

python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmedoids_index_k_512_m_147.db --codebook_file my_kmedoids_codebook_k_512_m_147.pkl > Kmedoids_pq_index_100_k_512_m_147.txt

# ------------------------
# Set 5: m = 1, num_clusters = 1024
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 147 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_147.pkl --pq_index_out my_kmedoids_index_k_1024_m_147.db

python queryfirstimplementation.py --m 147 --k 100 --db_file my_kmedoids_index_k_1024_m_147.db --codebook_file my_kmedoids_codebook_k_1024_m_147.pkl > Kmedoids_pq_index_100_k_1024_m_147.txt


python PQfirstImplementation_Kmedoids.py --m 1 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_1.pkl --pq_index_out my_kmedoids_index_k_256_m_1.db

python queryfirstimplementation.py --m 1 --k 100 --db_file my_kmedoids_index_k_256_m_1.db --codebook_file my_kmedoids_codebook_k_256_m_1.pkl > Kmedoids_pq_index_100_k_256_m_1.txt

# ------------------------
# Set 4: m = 1, num_clusters = 512
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 1 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_1.pkl --pq_index_out my_kmedoids_index_k_512_m_1.db

python queryfirstimplementation.py --m 1 --k 100 --db_file my_kmedoids_index_k_512_m_1.db --codebook_file my_kmedoids_codebook_k_512_m_1.pkl > Kmedoids_pq_index_100_k_512_m_1.txt

# ------------------------
# Set 5: m = 1, num_clusters = 1024
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 1 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_1.pkl --pq_index_out my_kmedoids_index_k_1024_m_1.db

python queryfirstimplementation.py --m 1 --k 100 --db_file my_kmedoids_index_k_1024_m_1.db --codebook_file my_kmedoids_codebook_k_1024_m_1.pkl > Kmedoids_pq_index_100_k_1024_m_1.txt


# ------------------------

python PQfirstImplementation_Kmedoids.py --m 49 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_49.pkl --pq_index_out my_kmedoids_index_k_256_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmedoids_index_k_256_m_49.db --codebook_file my_kmedoids_codebook_k_256_m_49.pkl > Kmedoids_pq_index_100_k_256_m_49.txt

# ------------------------
# Set 4: m = 1, num_clusters = 512
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 49 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_49.pkl --pq_index_out my_kmedoids_index_k_512_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmedoids_index_k_512_m_49.db --codebook_file my_kmedoids_codebook_k_512_m_49.pkl > Kmedoids_pq_index_100_k_512_m_49.txt

# ------------------------
# Set 5: m = 1, num_clusters = 1024
# ------------------------
python PQfirstImplementation_Kmedoids.py --m 49 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_49.pkl --pq_index_out my_kmedoids_index_k_1024_m_49.db

python queryfirstimplementation.py --m 49 --k 100 --db_file my_kmedoids_index_k_1024_m_49.db --codebook_file my_kmedoids_codebook_k_1024_m_49.pkl > Kmedoids_pq_index_100_k_1024_m_49.txt
