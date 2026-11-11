-- Step 3 of `python -m yrec.data build`, run once per cutoff ({cut} = A or AB); features for the ranker.
-- User, track and user x artist features as of {cutoff}. Only events before the cutoff are used.
-- Writes ufeat_{cut}.parquet, ifeat_{cut}.parquet and uartist_{cut}.parquet.

COPY (
    WITH l AS (
        SELECT uidx,
               count(*)                                         AS u_n_listen,
               sum(lp::INT)                                     AS u_n_lp,
               avg((NOT lp)::INT)                               AS u_skip_rate,
               avg(is_organic)                                  AS u_organic_share,
               1 - count(DISTINCT item_idx) / count(*)          AS u_repeat_ratio,
               count(DISTINCT artist_idx)                       AS u_n_artists,
               avg(len)                                         AS u_mean_len,
               ({cutoff} - max(ts)) / 86400.0                   AS u_days_since_last,
               (max(ts) - min(ts)) / 86400.0                    AS u_days_span,
               count(*) FILTER (WHERE ts >= {cutoff} - 7 * 86400) AS u_n_7d
        FROM listens WHERE ts < {cutoff} GROUP BY uidx),
    r AS (SELECT uidx, count(*) FILTER (WHERE ev = 'like') AS u_n_likes,
                 count(*) FILTER (WHERE ev = 'dislike') AS u_n_dislikes
          FROM reactions WHERE ts < {cutoff} GROUP BY uidx)
    SELECT l.*, coalesce(u_n_likes, 0) AS u_n_likes, coalesce(u_n_dislikes, 0) AS u_n_dislikes
    FROM l LEFT JOIN r USING (uidx)
) TO '{proc}/ufeat_{cut}.parquet';

COPY (
    WITH l AS (
        SELECT item_idx, any_value(artist_idx) AS artist_idx,
               count(*)                                            AS i_n_listen,
               sum(lp::INT)                                        AS i_n_lp,
               avg((NOT lp)::INT)                                  AS i_skip_rate,
               count(DISTINCT uidx)                                AS i_n_users,
               count(*) FILTER (WHERE ts >= {cutoff} - 86400)      AS i_pop_1d,
               count(*) FILTER (WHERE ts >= {cutoff} - 7 * 86400)  AS i_pop_7d,
               count(*) FILTER (WHERE ts >= {cutoff} - 30 * 86400) AS i_pop_30d,
               median(len)                                         AS i_len,
               ({cutoff} - min(ts)) / 86400.0                      AS i_age_days,
               avg(is_organic)                                     AS i_organic_share
        FROM listens WHERE ts < {cutoff} GROUP BY item_idx),
    r AS (SELECT item_idx, count(*) FILTER (WHERE ev = 'like') AS i_n_likes,
                 count(*) FILTER (WHERE ev = 'dislike') AS i_n_dislikes
          FROM reactions WHERE ts < {cutoff} GROUP BY item_idx)
    SELECT l.* EXCLUDE (artist_idx),
           i_pop_7d / (i_pop_30d / 4.0 + 1)                       AS i_trend,
           coalesce(i_n_likes, 0) / i_n_users                      AS i_like_rate,
           coalesce(i_n_dislikes, 0) / i_n_users                   AS i_dislike_rate,
           sum(i_n_lp) OVER (PARTITION BY artist_idx)              AS i_artist_n_lp
    FROM l LEFT JOIN r USING (item_idx)
) TO '{proc}/ifeat_{cut}.parquet';

COPY (
    SELECT uidx, artist_idx, count(*) AS ua_n_listen, sum(lp::INT) AS ua_n_lp
    FROM listens WHERE ts < {cutoff} GROUP BY ALL
) TO '{proc}/uartist_{cut}.parquet';
