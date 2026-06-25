"""Phyling API connector and API-level caches."""

import io
import os
import pickle
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from config import (
    API_RECORDS_CACHE,
    API_RECORDS_CACHE_TTL_S,
    CACHE_DIR,
    PHYLING_BASE_URL,
    PHYLING_CLIENT_ID,
)


def has_api_key(api_key):
    """Return True when the Phyling API can be queried."""
    return bool(api_key)


def clear_phyling_disk_cache():
    """Clear persisted Phyling metadata cache."""
    try:
        if os.path.exists(API_RECORDS_CACHE):
            os.remove(API_RECORDS_CACHE)
    except Exception:
        pass


def _phyling_headers(api_key):
    if not api_key:
        return None
    return {
        "Authorization": f"ApiKey {api_key}",
        "Content-Type": "application/json",
    }


def _read_phyling_disk_cache():
    """Read persisted Phyling metadata when it is still fresh."""
    try:
        if not os.path.exists(API_RECORDS_CACHE):
            return None
        if time.time() - os.path.getmtime(API_RECORDS_CACHE) > API_RECORDS_CACHE_TTL_S:
            return None
        with open(API_RECORDS_CACHE, "rb") as f:
            payload = pickle.load(f)
        if payload.get("client_id") != PHYLING_CLIENT_ID:
            return None
        records = payload.get("records", [])
        status = dict(payload.get("status", {}))
        status["message"] = status.get("message", "Cache Phyling")
        status["from_disk_cache"] = True
        return records, status
    except Exception:
        return None


def _write_phyling_disk_cache(records, status):
    """Persist Phyling metadata for faster cold starts."""
    try:
        payload = {
            "client_id": PHYLING_CLIENT_ID,
            "records": records,
            "status": status,
        }
        with open(API_RECORDS_CACHE, "wb") as f:
            pickle.dump(payload, f)
    except Exception:
        pass


def _record_selections(rec):
    """
    Return selections for a Phyling record.
    Some /records/all responses do not embed selections; in that case expose
    the full record as one loadable item and fetch decoded CSV without sel_id.
    """
    for key in ("selections", "selection", "selected_parts", "parts"):
        value = rec.get(key)
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict) and value:
            return [value]

    sel_id = rec.get("sel_id") or rec.get("selection_id")
    return [{
        "id": sel_id,
        "num": rec.get("sel") or rec.get("selection_num") or 1,
        "comment": rec.get("comment", ""),
        "exercise_name": rec.get("exercise_name", ""),
        "_full_record": True,
    }]


def _empty_status(api_key):
    return {
        "enabled": has_api_key(api_key),
        "ok": False,
        "pages": 0,
        "total_api": None,
        "raw_records": 0,
        "kayak_selections": 0,
        "records_with_selections": 0,
        "full_record_fallbacks": 0,
        "sports_seen": [],
        "groups_seen": [],
        "message": "",
    }


def fetch_phyling_records(api_key, page_size=500, days_back=None):
    records, _status = fetch_phyling_records_with_status(
        api_key=api_key,
        page_size=page_size,
        days_back=days_back,
    )
    return records


def fetch_phyling_records_with_status(api_key, page_size=500, days_back=None):
    """
    Fetch Phyling records and return (records, status).
    status is intentionally UI-friendly for sidebar diagnostics.
    """
    status = _empty_status(api_key)

    if days_back is None:
        cached = _read_phyling_disk_cache()
        if cached is not None:
            return cached

    headers = _phyling_headers(api_key)
    if not headers:
        status["message"] = "PHYLING_API_KEY absente"
        return [], status

    all_records = []
    page = 1
    cutoff = ((datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
              if days_back else None)

    while True:
        try:
            response = requests.post(
                f"{PHYLING_BASE_URL}/records/all",
                headers=headers,
                json={
                    "type": "associated",
                    "pageId": page,
                    "pageSize": page_size,
                    "clientIds": [PHYLING_CLIENT_ID],
                    "userIds": [],
                    "deviceIds": [],
                    "groupIds": [],
                    "exerciseIds": [],
                    "onlyFavorite": False,
                },
                timeout=30,
            )
        except Exception as exc:
            status["message"] = f"Erreur API: {exc.__class__.__name__}"
            break

        if response.status_code != 200:
            status["message"] = f"Erreur API HTTP {response.status_code}"
            break

        data = response.json()
        records = data.get("records", [])
        status["pages"] = page
        status["total_api"] = data.get("total", status["total_api"])
        status["raw_records"] += len(records)
        if not records:
            status["message"] = "API OK, aucun record retourné"
            break

        stop_pagination = False
        for rec in records:
            sport_name = str(rec.get("sport_name", "") or "")
            group_name = str(rec.get("group_name", "") or "")
            if sport_name and sport_name not in status["sports_seen"]:
                status["sports_seen"].append(sport_name)
            if group_name and group_name not in status["groups_seen"]:
                status["groups_seen"].append(group_name)

            selections = _record_selections(rec)
            if not selections:
                continue
            status["records_with_selections"] += 1

            try:
                rec_dt = datetime.strptime(rec["date"], "%d/%m/%Y %H:%M:%S")
                rec_date = rec_dt.strftime("%Y-%m-%d")
                heure_str = rec_dt.strftime("%H:%M")
            except Exception:
                rec_date = rec.get("date", "")[:10]
                heure_str = ""

            if cutoff and rec_date < cutoff:
                stop_pagination = True
                continue

            users = rec.get("users", [])
            athlete = " ".join([
                u.get("firstname", "").capitalize() + " " +
                u.get("lastname", "").upper()
                for u in users
            ]).strip() if users else "Inconnu"

            other = {}
            try:
                import json as _json
                other = _json.loads(rec.get("other_data", "{}") or "{}")
            except Exception:
                pass
            bateau = other.get("boat", "").upper()

            group = group_name
            group_map = {
                "Kayak_D": ("Kayak", "F"), "Kayak_H": ("Kayak", "H"),
                "Canoe_D": ("Canoë", "F"), "Canoe_H": ("Canoë", "H"),
            }
            discipline, sexe = group_map.get(group, ("Kayak", ""))

            comment_g = str(rec.get("comment", "") or "").lower()
            exercise_g = str(rec.get("exercise_name", "") or "").lower()
            competition_words = {"fa", "fb", "finale", "sf", "demi", "serie",
                                 "course", "race", "championnat"}
            is_competition = any(k in comment_g or k in exercise_g
                                 for k in competition_words)

            for sel in selections:
                sel_id = sel.get("id")
                sel_num = sel.get("num", 1)
                if sel.get("_full_record"):
                    status["full_record_fallbacks"] += 1
                comment = str(sel.get("comment", rec.get("comment", "")) or "")
                ex_name = str(sel.get("exercise_name", rec.get("exercise_name", "")) or "")

                dist = ""
                for text in [comment, ex_name, comment_g]:
                    match = re.search(r"(\d+)\s*m", text, re.IGNORECASE)
                    if match and match.group(1) in {"200", "250", "500", "1000", "2000"}:
                        dist = match.group(1) + "m"
                        break

                sel_is_competition = any(k in comment.lower() for k in competition_words)
                type_course = "Compétition" if (is_competition or sel_is_competition) else "Entraînement"

                all_records.append({
                    "fichier": f"{rec['id']}:{sel_id or ''}",
                    "athlete": athlete,
                    "distance": dist,
                    "date": rec_date,
                    "heure": heure_str,
                    "sel": str(sel_num),
                    "notes": comment,
                    "discipline": discipline,
                    "sexe": sexe,
                    "categorie": "",
                    "bateau": bateau,
                    "type_course": type_course,
                    "lieu": "",
                    "rec_id": rec["id"],
                    "sel_id": sel_id,
                    "group_name": group,
                })
                status["kayak_selections"] += 1

        total = data.get("total", 0)
        if stop_pagination or page * page_size >= total:
            break
        page += 1

    if all_records:
        status["ok"] = True
        status["message"] = f"{len(all_records)} sélection(s) depuis Phyling"
        if days_back is None:
            _write_phyling_disk_cache(all_records, status)
    elif not status["message"]:
        sports = ", ".join(status["sports_seen"][:4]) or "sport_name vide"
        groups = ", ".join(status["groups_seen"][:4]) or "group_name vide"
        status["message"] = (
            "API OK, aucune sélection exploitable après filtrage "
            f"(sports: {sports}; groupes: {groups})"
        )

    return all_records, status


def fetch_csv_from_api(api_key, rec_id, sel_id):
    """Download one decoded Phyling CSV as a DataFrame."""
    sel_id = "" if sel_id is None else str(sel_id)
    cache_suffix = sel_id if sel_id else "full"
    cache_path = os.path.join(CACHE_DIR, f"{rec_id}_{cache_suffix}.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    headers = _phyling_headers(api_key)
    if not headers:
        return pd.DataFrame()

    params = {"sel_id": sel_id} if sel_id else {}
    response = requests.post(
        f"{PHYLING_BASE_URL}/records/{rec_id}/file/decoded",
        headers=headers,
        json={},
        params=params,
        timeout=120,
    )
    if response.status_code != 200 and sel_id:
        response = requests.post(
            f"{PHYLING_BASE_URL}/records/{rec_id}/file/decoded",
            headers=headers,
            json={},
            params={},
            timeout=120,
        )
    if response.status_code != 200:
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(response.text))
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass
    return df

