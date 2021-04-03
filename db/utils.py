import io
from tqdm import tqdm


def chunker(seq, size):
    # http://stackoverflow.com/a/434328
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def insert_with_progress(df, conn, table, chunks=10, pbar_desc=None, sep=','):
    chunksize = int(len(df) / chunks)
    desc = 'Insert data'

    if pbar_desc is not None:
        desc += f' ({pbar_desc})'

    with tqdm(total=len(df), desc=desc, leave=False) as pbar:
        for cdf in chunker(df, chunksize):
            cursor = conn.cursor()
            fbuf = io.StringIO()
            cdf.to_csv(fbuf, index=False, header=False, sep=sep)
            fbuf.seek(0)
            cursor.copy_from(fbuf, table, sep=sep, null='')
            conn.commit()
            cursor.close()
            pbar.update(chunksize)
