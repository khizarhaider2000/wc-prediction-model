"""Build a 2026-style Round of 32 bracket from group-stage results."""


ROUND_OF_32_SLOT_TEMPLATE = [
    ("1A", "3CDEFG"),
    ("2A", "2B"),
    ("1B", "3CDEFG"),
    ("1C", "3ABFHI"),
    ("2C", "2D"),
    ("1D", "3BEFIJK"),
    ("1E", "3ABCDL"),
    ("2E", "2F"),
    ("1F", "3ABCDE"),
    ("1G", "3ABCDE"),
    ("2G", "2H"),
    ("1H", "2I"),
    ("1I", "2L"),
    ("1J", "2K"),
    ("1K", "3ABCDF"),
    ("1L", "2J"),
]

ALL_GROUPS = list("ABCDEFGHIJKL")


def parse_slot_code(slot_code: str) -> dict:
    """
    Parses a bracket slot code into its qualification type and allowed groups.

    Examples:
    - "1A" -> {"type": "winner", "groups": ["A"]}
    - "2B" -> {"type": "runner_up", "groups": ["B"]}
    - "3CDEFG" -> {"type": "third_place", "groups": ["C", "D", "E", "F", "G"]}
    """

    if len(slot_code) < 2:
        raise ValueError(f"Invalid slot code '{slot_code}'.")

    slot_type_code = slot_code[0]
    groups = list(slot_code[1:].upper())

    slot_types = {
        "1": "winner",
        "2": "runner_up",
        "3": "third_place",
    }

    if slot_type_code not in slot_types:
        raise ValueError(f"Invalid slot code '{slot_code}'. Expected 1, 2, or 3.")

    if not groups:
        raise ValueError(f"Invalid slot code '{slot_code}'. Missing group letters.")

    return {
        "type": slot_types[slot_type_code],
        "groups": groups,
    }


def get_team_group_map(group_stage: dict) -> dict[str, str]:
    """
    Builds a team-to-group map from the simulated group tables.
    """

    group_tables = group_stage.get("group_tables", {})
    team_group_map = {}

    for group, table in group_tables.items():
        for row in table:
            team_group_map[row["team"]] = group

    return team_group_map


def get_group_position_maps(group_stage: dict) -> tuple[dict[str, str], dict[str, str]]:
    """
    Returns lookup maps for group winners and runners-up by group letter.
    """

    group_tables = group_stage.get("group_tables", {})
    group_winners_by_group = {}
    runners_up_by_group = {}

    for group, table in group_tables.items():
        if len(table) < 2:
            raise ValueError(f"Group {group} table must contain at least two teams.")

        group_winners_by_group[group] = table[0]["team"]
        runners_up_by_group[group] = table[1]["team"]

    return group_winners_by_group, runners_up_by_group


def assign_third_place_teams_to_slots(
    third_place_slots: list[str],
    best_third_place: list[dict],
    fixed_opponents: list[str],
    team_group_map: dict[str, str],
) -> dict[int, str]:
    """
    Assigns qualified third-place teams to bracket slots with backtracking.

    Returns a mapping from the third-place slot list index to the assigned team.
    """

    if len(third_place_slots) != len(fixed_opponents):
        raise ValueError("Third-place slots and fixed opponents must have equal length.")

    if len(third_place_slots) != len(best_third_place):
        raise ValueError(
            "Third-place slot count must match qualified third-place team count. "
            f"Got {len(third_place_slots)} slots and {len(best_third_place)} teams."
        )

    candidates_by_slot = {}

    for slot_index, slot_code in enumerate(third_place_slots):
        parsed_slot = parse_slot_code(slot_code)
        allowed_groups = set(parsed_slot["groups"])
        opponent_group = team_group_map.get(fixed_opponents[slot_index])
        candidates = []

        for row in best_third_place:
            team = row["team"]
            group = row["group"]

            if group not in allowed_groups:
                continue

            if opponent_group is not None and group == opponent_group:
                continue

            candidates.append(team)

        candidates_by_slot[slot_index] = candidates

    slot_order = sorted(
        range(len(third_place_slots)),
        key=lambda index: (len(candidates_by_slot[index]), index),
    )
    team_strength_order = {
        row["team"]: index
        for index, row in enumerate(best_third_place)
    }
    assignment = {}
    used_teams = set()

    def backtrack(order_index: int) -> bool:
        if order_index == len(slot_order):
            return True

        slot_index = slot_order[order_index]
        candidates = sorted(
            candidates_by_slot[slot_index],
            key=lambda team: team_strength_order[team],
        )

        for team in candidates:
            if team in used_teams:
                continue

            assignment[slot_index] = team
            used_teams.add(team)

            if backtrack(order_index + 1):
                return True

            used_teams.remove(team)
            del assignment[slot_index]

        return False

    if backtrack(0):
        return assignment

    qualified_third_groups = [row["group"] for row in best_third_place]
    slot_details = [
        {
            "slot_index": index,
            "slot_code": slot,
            "allowed_groups": parse_slot_code(slot)["groups"],
            "fixed_opponent": fixed_opponents[index],
            "fixed_opponent_group": team_group_map.get(fixed_opponents[index]),
            "eligible_teams": candidates_by_slot[index],
        }
        for index, slot in enumerate(third_place_slots)
    ]

    raise ValueError(
        "No valid third-place bracket assignment found. "
        f"Qualified third-place groups: {qualified_third_groups}. "
        f"Third-place slots needed: {third_place_slots}. "
        f"Slot eligibility after same-group rematch checks: {slot_details}."
    )


def validate_matchups(
    matchups: list[tuple[str, str]],
    qualified_teams: list[str],
) -> None:
    """
    Validates the completed Round of 32 matchups.
    """

    if len(matchups) != 16:
        raise ValueError(f"Round of 32 must contain 16 matchups. Got {len(matchups)}.")

    teams = [team for matchup in matchups for team in matchup]

    if len(teams) != 32:
        raise ValueError(f"Round of 32 must contain 32 teams. Got {len(teams)}.")

    duplicate_teams = sorted({team for team in teams if teams.count(team) > 1})

    if duplicate_teams:
        raise ValueError(f"Duplicate teams in Round of 32: {duplicate_teams}.")

    qualified_team_set = set(qualified_teams)
    matchup_team_set = set(teams)
    unexpected_teams = sorted(matchup_team_set - qualified_team_set)
    missing_teams = sorted(qualified_team_set - matchup_team_set)

    if unexpected_teams:
        raise ValueError(f"Round of 32 contains unqualified teams: {unexpected_teams}.")

    if missing_teams:
        raise ValueError(f"Qualified teams missing from Round of 32: {missing_teams}.")

    self_matches = [
        matchup
        for matchup in matchups
        if matchup[0] == matchup[1]
    ]

    if self_matches:
        raise ValueError(f"Team cannot play itself in Round of 32: {self_matches}.")


def create_round_of_32_matchups(group_stage: dict) -> list[tuple[str, str]]:
    """
    Creates 2026-style Round of 32 matchups from a group-stage simulation.
    """

    team_group_map = get_team_group_map(group_stage)
    group_winners_by_group, runners_up_by_group = get_group_position_maps(group_stage)

    resolved_slots: list[list[str | None]] = []
    third_place_slot_indexes = []
    third_place_slots = []
    fixed_opponents = []

    for matchup_index, matchup_slots in enumerate(ROUND_OF_32_SLOT_TEMPLATE):
        resolved_matchup = []

        for slot_position, slot_code in enumerate(matchup_slots):
            parsed_slot = parse_slot_code(slot_code)
            group = parsed_slot["groups"][0]

            if parsed_slot["type"] == "winner":
                resolved_matchup.append(group_winners_by_group[group])
            elif parsed_slot["type"] == "runner_up":
                resolved_matchup.append(runners_up_by_group[group])
            else:
                resolved_matchup.append(None)
                fixed_position = 1 - slot_position
                fixed_opponent_slot = matchup_slots[fixed_position]
                fixed_parsed_slot = parse_slot_code(fixed_opponent_slot)
                fixed_group = fixed_parsed_slot["groups"][0]

                if fixed_parsed_slot["type"] == "winner":
                    fixed_opponent = group_winners_by_group[fixed_group]
                elif fixed_parsed_slot["type"] == "runner_up":
                    fixed_opponent = runners_up_by_group[fixed_group]
                else:
                    raise ValueError(
                        "Third-place slots cannot be paired with another "
                        f"third-place slot. Matchup: {matchup_slots}."
                    )

                third_place_slot_indexes.append((matchup_index, slot_position))
                third_place_slots.append(slot_code)
                fixed_opponents.append(fixed_opponent)

        resolved_slots.append(resolved_matchup)

    try:
        third_place_assignments = assign_third_place_teams_to_slots(
            third_place_slots=third_place_slots,
            best_third_place=group_stage["best_third_place"],
            fixed_opponents=fixed_opponents,
            team_group_map=team_group_map,
        )
    except ValueError as strict_error:
        expanded_third_place_slots = [
            f"3{''.join(ALL_GROUPS)}"
            for _ in third_place_slots
        ]

        try:
            third_place_assignments = assign_third_place_teams_to_slots(
                third_place_slots=expanded_third_place_slots,
                best_third_place=group_stage["best_third_place"],
                fixed_opponents=fixed_opponents,
                team_group_map=team_group_map,
            )
        except ValueError as expanded_error:
            raise ValueError(
                "No valid third-place bracket assignment found after trying "
                "restricted and expanded slot pools. "
                f"Restricted assignment failure: {strict_error}. "
                f"Expanded assignment failure: {expanded_error}."
            ) from expanded_error

    for slot_index, team in third_place_assignments.items():
        matchup_index, slot_position = third_place_slot_indexes[slot_index]
        resolved_slots[matchup_index][slot_position] = team

    matchups = [
        (team_a, team_b)
        for team_a, team_b in resolved_slots
        if team_a is not None and team_b is not None
    ]

    validate_matchups(matchups, group_stage["qualified_teams"])

    return matchups


def debug_print_round_of_32_bracket(group_stage: dict | None = None) -> None:
    """
    Prints one simulated Round of 32 bracket for inspection.
    """

    if group_stage is None:
        from src.simulation.simulate_group_stage import simulate_full_group_stage_once

        group_stage = simulate_full_group_stage_once()

    matchups = create_round_of_32_matchups(group_stage)

    print("\nRound of 32 Matchups\n")

    for team_a, team_b in matchups:
        print(f"{team_a} vs {team_b}")
