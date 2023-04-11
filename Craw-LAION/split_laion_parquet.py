import os
import pyarrow.parquet as pq
import tqdm
import fastparquet as fp

parquet_path = '/home/leiningjie/PycharmProjects/dataset/LAION/LAION-2B-en/0/part-00000-5114fd87-297e-42b0-9d11-50f1df323dfa-c000.snappy.parquet'

fold = os.path.dirname(parquet_path)
if os.path.exists(f'{fold}/split') is False:
    os.makedirs(f'{fold}/split')

pq_data = pq.read_table(parquet_path, columns=['SAMPLE_ID','URL'])
df = pq_data.to_pandas()

# 每批数据的行数
batch_size = 20000

# 计算总批数
total_batches = len(df) // batch_size + 1

# 拆分并保存每个批次的数据到不同的 Parquet 文件中
for batch_num in range(total_batches):
    start_index = batch_num * batch_size
    end_index = min((batch_num + 1) * batch_size, len(df))
    batch_df = df.iloc[start_index:end_index]
    fp.write(f'{fold}/split/batch-{batch_num}.parquet', batch_df)