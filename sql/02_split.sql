-- History before {cutoff} and labels for the window [{wstart}, {wend}) that follows it.
-- Nothing at or after {cutoff} feeds the history table.

CREATE OR REPLACE TABLE hist AS
WITH l AS (
    SELECT uidx, item_idx, count(*) AS n_listen, sum(lp::INT) AS n_lp,
           max(ts) AS last_ts, avg(ratio) AS avg_ratio
    FROM listens WHERE ts < {cutoff} GROUP BY ALL),
r AS (  -- final like / dislike state as of the cutoff (unlike / undislike revert it)
    SELECT uidx, item_idx,
           arg_max(ev, ts) FILTER (WHERE ev IN ('like', 'unlike'))       = 'like'    AS liked,
           arg_max(ev, ts) FILTER (WHERE ev IN ('dislike', 'undislike')) = 'dislike' AS disliked
    FROM reactions WHERE ts < {cutoff} GROUP BY ALL)
SELECT uidx, item_idx,
       coalesce(n_listen, 0) AS n_listen, coalesce(n_lp, 0) AS n_lp, last_ts, avg_ratio,
       coalesce(liked, FALSE) AS liked, coalesce(disliked, FALSE) AS disliked
FROM l FULL JOIN r USING (uidx, item_idx);

COPY hist TO '{proc}/hist_{cut}.parquet';

COPY (
    WITH l AS (SELECT uidx, item_idx, bool_or(lp) AS lp
               FROM listens WHERE ts >= {wstart} AND ts < {wend} GROUP BY ALL),
    r AS (SELECT uidx, item_idx, bool_or(ev = 'like') AS liked, bool_or(ev = 'dislike') AS disliked
          FROM reactions WHERE ts >= {wstart} AND ts < {wend} GROUP BY ALL)
    SELECT uidx, item_idx,
           coalesce(l.lp, FALSE) AS lp, coalesce(r.liked, FALSE) AS liked, coalesce(r.disliked, FALSE) AS disliked,
           coalesce(h.n_listen, 0) = 0 AS is_new                 -- never played before the cutoff
    FROM l FULL JOIN r USING (uidx, item_idx)
    LEFT JOIN hist h USING (uidx, item_idx)
    WHERE uidx IN (SELECT DISTINCT uidx FROM hist)              -- no cold users, as in the official split
) TO '{proc}/tgt_{cut}.parquet';
