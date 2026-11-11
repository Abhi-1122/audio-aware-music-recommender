-- Raw Yambda events -> typed tables with dense user / item / artist indices.
CREATE TABLE listens_raw AS SELECT * FROM read_parquet('{raw}/listens.parquet');

CREATE TABLE reactions_raw AS
          SELECT uid, item_id, timestamp AS ts, 'like'      AS ev FROM read_parquet('{raw}/likes.parquet')
UNION ALL SELECT uid, item_id, timestamp,       'unlike'          FROM read_parquet('{raw}/unlikes.parquet')
UNION ALL SELECT uid, item_id, timestamp,       'dislike'         FROM read_parquet('{raw}/dislikes.parquet')
UNION ALL SELECT uid, item_id, timestamp,       'undislike'       FROM read_parquet('{raw}/undislikes.parquet');

CREATE TABLE users AS
SELECT (row_number() OVER (ORDER BY uid) - 1)::INT AS uidx, uid
FROM (SELECT uid FROM listens_raw UNION SELECT uid FROM reactions_raw);

-- A track can list several artists/albums; keep the smallest id. Unknown artist -> last index.
CREATE TABLE items AS
WITH ids AS (SELECT item_id FROM listens_raw UNION SELECT item_id FROM reactions_raw),
ar AS (SELECT item_id, min(artist_id) AS artist_id FROM read_parquet('{raw}/artist_item_mapping.parquet') GROUP BY 1),
al AS (SELECT item_id, min(album_id)  AS album_id  FROM read_parquet('{raw}/album_item_mapping.parquet')  GROUP BY 1)
SELECT (row_number() OVER (ORDER BY item_id) - 1)::INT                    AS idx,
       item_id,
       (dense_rank() OVER (ORDER BY artist_id NULLS LAST) - 1)::INT       AS artist_idx,
       (dense_rank() OVER (ORDER BY album_id  NULLS LAST) - 1)::INT       AS album_idx
FROM ids LEFT JOIN ar USING (item_id) LEFT JOIN al USING (item_id);

CREATE TABLE listens AS
SELECT u.uidx, i.idx AS item_idx, i.artist_idx, l.timestamp::BIGINT AS ts, l.is_organic,
       l.played_ratio_pct AS ratio, l.track_length_seconds AS len,
       l.played_ratio_pct >= 50 AS lp                                     -- "Listen+"
FROM listens_raw l JOIN users u USING (uid) JOIN items i USING (item_id);

CREATE TABLE reactions AS
SELECT u.uidx, i.idx AS item_idx, r.ts::BIGINT AS ts, r.ev
FROM reactions_raw r JOIN users u USING (uid) JOIN items i USING (item_id);

DROP TABLE listens_raw; DROP TABLE reactions_raw;

COPY users TO '{proc}/users.parquet';
COPY (SELECT uidx, item_idx, ts FROM listens WHERE lp ORDER BY uidx, ts) TO '{proc}/lp_events.parquet';
