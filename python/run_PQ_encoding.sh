# #!/bin/bash

python PQfirstImplementation_kmeans.py --m 14 --num_clusters 256 --codebook_out my_kmeans_codebook_k_256_m_14.pkl --pq_index_out my_kmeans_index_k_256_m_14.db

python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmeans_index_k_256_m_14.db --codebook_file my_kmeans_codebook_k_256_m_14.pkl > kmeans_pq_index_100_k_256_m_14.txt

# ------------------------
# Set 1: m = 14, num_clusters = 512
# ------------------------
python PQfirstImplementation_kmeans.py --m 14 --num_clusters 512 --codebook_out my_kmeans_codebook_k_512_m_14.pkl --pq_index_out my_kmeans_index_k_512_m_14.db

python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmeans_index_k_512_m_14.db --codebook_file my_kmeans_codebook_k_512_m_14.pkl > kmeans_pq_index_100_k_512_m_14.txt

# ------------------------
# Set 2: m = 14, num_clusters = 1024
# ------------------------
python PQfirstImplementation_kmeans.py --m 14 --num_clusters 1024 --codebook_out my_kmeans_codebook_k_1024_m_14.pkl --pq_index_out my_kmeans_index_k_1024_m_14.db

python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmeans_index_k_1024_m_14.db --codebook_file my_kmeans_codebook_k_1024_m_14.pkl > kmeans_pq_index_100_k_1024_m_14.txt

# ------------------------
# Set 3: m = 26, num_clusters = 256
# ------------------------
python PQfirstImplementation_kmeans.py --m 26 --num_clusters 256 --codebook_out my_kmeans_codebook_k_256_m_26.pkl --pq_index_out my_kmeans_index_k_256_m_26.db

python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmeans_index_k_256_m_26.db --codebook_file my_kmeans_codebook_k_256_m_26.pkl > kmeans_pq_index_100_k_256_m_26.txt

# ------------------------
# Set 4: m = 26, num_clusters = 512
# ------------------------
python PQfirstImplementation_kmeans.py --m 26 --num_clusters 512 --codebook_out my_kmeans_codebook_k_512_m_26.pkl --pq_index_out my_kmeans_index_k_512_m_26.db

python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmeans_index_k_512_m_26.db --codebook_file my_kmeans_codebook_k_512_m_26.pkl > kmeans_pq_index_100_k_512_m_26.txt

# ------------------------
# Set 5: m = 26, num_clusters = 1024
# ------------------------
python PQfirstImplementation_kmeans.py --m 26 --num_clusters 1024 --codebook_out my_kmeans_codebook_k_1024_m_26.pkl --pq_index_out my_kmeans_index_k_1024_m_26.db

python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmeans_index_k_1024_m_26.db --codebook_file my_kmeans_codebook_k_1024_m_26.pkl > kmeans_pq_index_100_k_1024_m_26.txt

# ------------------------
# Set 3: m = 202, num_clusters = 256
# ------------------------
python PQfirstImplementation_kmeans.py --m 202 --num_clusters 256 --codebook_out my_kmeans_codebook_k_256_m_202.pkl --pq_index_out my_kmeans_index_k_256_m_202.db

python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmeans_index_k_256_m_202.db --codebook_file my_kmeans_codebook_k_256_m_202.pkl > kmeans_pq_index_100_k_256_m_202.txt

# ------------------------
# Set 4: m = 202, num_clusters = 512
# ------------------------
python PQfirstImplementation_kmeans.py --m 202 --num_clusters 512 --codebook_out my_kmeans_codebook_k_512_m_202.pkl --pq_index_out my_kmeans_index_k_512_m_202.db

python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmeans_index_k_512_m_202.db --codebook_file my_kmeans_codebook_k_512_m_202.pkl > kmeans_pq_index_100_k_512_m_202.txt

# ------------------------
# Set 5: m = 202, num_clusters = 1024
# ------------------------
python PQfirstImplementation_kmeans.py --m 202 --num_clusters 1024 --codebook_out my_kmeans_codebook_k_1024_m_202.pkl --pq_index_out my_kmeans_index_k_1024_m_202.db

python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmeans_index_k_1024_m_202.db --codebook_file my_kmeans_codebook_k_1024_m_202.pkl > kmeans_pq_index_100_k_1024_m_202.txt

# python PQfirstImplementation_Kmedoids.py --m 14 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_14.pkl --pq_index_out my_kmedoids_index_k_256_m_14.db

# python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmedoids_index_k_256_m_14.db --codebook_file my_kmedoids_codebook_k_256_m_14.pkl > Kmedoids_pq_index_100_k_256_m_14.txt

# # ------------------------
# # Set 1: m = 14, num_clusters = 512
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 14 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_14.pkl --pq_index_out my_kmedoids_index_k_512_m_14.db

# python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmedoids_index_k_512_m_14.db --codebook_file my_kmedoids_codebook_k_512_m_14.pkl > Kmedoids_pq_index_100_k_512_m_14.txt

# # ------------------------
# # Set 2: m = 14, num_clusters = 1024
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 14 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_14.pkl --pq_index_out my_kmedoids_index_k_1024_m_14.db

# python queryfirstimplementation.py --m 14 --k 100 --db_file my_kmedoids_index_k_1024_m_14.db --codebook_file my_kmedoids_codebook_k_1024_m_14.pkl > Kmedoids_pq_index_100_k_1024_m_14.txt

# # ------------------------
# # Set 3: m = 26, num_clusters = 256
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 26 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_26.pkl --pq_index_out my_kmedoids_index_k_256_m_26.db

# python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmedoids_index_k_256_m_26.db --codebook_file my_kmedoids_codebook_k_256_m_26.pkl > Kmedoids_pq_index_100_k_256_m_26.txt

# # ------------------------
# # Set 4: m = 26, num_clusters = 512
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 26 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_26.pkl --pq_index_out my_kmedoids_index_k_512_m_26.db

# python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmedoids_index_k_512_m_26.db --codebook_file my_kmedoids_codebook_k_512_m_26.pkl > Kmedoids_pq_index_100_k_512_m_26.txt

# # ------------------------
# # Set 5: m = 26, num_clusters = 1024
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 26 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_26.pkl --pq_index_out my_kmedoids_index_k_1024_m_26.db

# python queryfirstimplementation.py --m 26 --k 100 --db_file my_kmedoids_index_k_1024_m_26.db --codebook_file my_kmedoids_codebook_k_1024_m_26.pkl > Kmedoids_pq_index_100_k_1024_m_26.txt


# # ------------------------
# # Set 3: m = 202, num_clusters = 256
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 202 --num_clusters 256 --codebook_out my_kmedoids_codebook_k_256_m_202.pkl --pq_index_out my_kmedoids_index_k_256_m_202.db

# python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmedoids_index_k_256_m_202.db --codebook_file my_kmedoids_codebook_k_256_m_202.pkl > Kmedoids_pq_index_100_k_256_m_202.txt

# # ------------------------
# # Set 4: m = 202, num_clusters = 512
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 202 --num_clusters 512 --codebook_out my_kmedoids_codebook_k_512_m_202.pkl --pq_index_out my_kmedoids_index_k_512_m_202.db

# python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmedoids_index_k_512_m_202.db --codebook_file my_kmedoids_codebook_k_512_m_202.pkl > Kmedoids_pq_index_100_k_512_m_202.txt

# # ------------------------
# # Set 5: m = 202, num_clusters = 1024
# # ------------------------
# python PQfirstImplementation_Kmedoids.py --m 202 --num_clusters 1024 --codebook_out my_kmedoids_codebook_k_1024_m_202.pkl --pq_index_out my_kmedoids_index_k_1024_m_202.db

# python queryfirstimplementation.py --m 202 --k 100 --db_file my_kmedoids_index_k_1024_m_202.db --codebook_file my_kmedoids_codebook_k_1024_m_202.pkl > Kmedoids_pq_index_100_k_1024_m_202.txt
