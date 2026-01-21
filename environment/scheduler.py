import random
from environment.actions import Action
import logging

logger = logging.getLogger("Scheduler")


class Scheduler:
    """
    Steuert die Reihenfolge der Agenten, die Memory-Grid-Daten
    (known, safe, risky), Reward-Shaping und Knowledge-Integration.
    """

    def __init__(self, agents, world, bus=None):
        self.agents = agents
        self.world = world
        self.turn = 0
        self.bus = bus

        # Knowledge-Sets
        self.known = set()   # Felder, die besucht wurden
        self.safe = set()    # Felder ohne Breeze/Stench
        self.risky = set()   # Felder mit Breeze/Stench
        # world.grid_size existiert bereits laut deinem Code

    # -----------------------------------------------------
    # Memory vor Bewegung aktualisieren
    # -----------------------------------------------------
    def _update_memory_before_move(self, agent, percepts):
        x, y = agent.pos()
        self.known.add((x, y))

        breeze = percepts.get("breeze", 0)
        stench = percepts.get("stench", 0)

        if breeze == 0 and stench == 0:
            self.safe.add((x, y))
            zone = "SAFE"
        else:
            self.risky.add((x, y))
            zone = "RISKY"

        logger.debug(
            f"_update_memory_before_move: agent={agent.role}, pos={(x, y)}, "
            f"breeze={breeze}, stench={stench}, zone={zone}"
        )


    # -----------------------------------------------------
    # Memory nach Bewegung aktualisieren
    # -----------------------------------------------------
    def _update_memory_after_move(self, agent, percepts):
        x, y = agent.pos()
        self.known.add((x, y))

        breeze = percepts.get("breeze", 0)
        stench = percepts.get("stench", 0)

        if breeze == 0 and stench == 0:
            self.safe.add((x, y))
            zone = "SAFE"
        else:
            self.risky.add((x, y))
            zone = "RISKY"

        logger.debug(
            f"_update_memory_after_move: agent={agent.role}, pos={(x, y)}, "
            f"breeze={breeze}, stench={stench}, zone={zone}"
        )

        #Nachricht senden wenn breeze 
        if breeze and self.bus:
            self.bus.broadcast(
                sender=agent.role,
                topic="Breeze_Detected",
                payload={"pos": (x,y), "breeze": breeze}
            )

        #Nachricht senden wenn stench
        if stench and self.bus:
            self.bus.broadcast(
                sender=agent.role,
                topic="Stench_Detected",
                payload={"pos": (x,y), "stench": stench}
            )


    # -----------------------------------------------------
    # Reward-Shaping
    # -----------------------------------------------------
    def _apply_reward_shaping(self, agent, old_pos, new_pos, percepts, new_percepts):
        if hasattr(agent, "reward"):
            before = agent.reward

            # Standard: -1 pro Schritt
            agent.reward -= 1

            if new_pos in self.risky:
                agent.reward -= 10

            if new_pos not in self.known:
                agent.reward += 5
            #!! es gibt kein glitter in percepts
            if new_percepts.get("glitter", 0) == 1:
                agent.reward += 100

            logger.debug(
                f"_apply_reward_shaping: agent={agent.role}, old_pos={old_pos}, "
                f"new_pos={new_pos}, reward_before={before}, reward_after={agent.reward}"
            )

    # -----------------------------------------------------
    # OBSERVATION für A1 (QMIX-Style)
    # -----------------------------------------------------
    def _build_qmix_observation(self, agent, percepts):
        x, y = agent.pos()
        grid = self.world.grid_size
        x_norm = x / (grid - 1)
        y_norm = y / (grid - 1)

        breeze = percepts.get("breeze", 0)
        stench = percepts.get("stench", 0)
        glitter = percepts.get("glitter", 0)

        known_fraction = len(self.known) / (grid * grid)

        obs = [
            x_norm,
            y_norm,
            float(breeze),
            float(stench),
            float(glitter),
            known_fraction,
        ]

        logger.debug(
            f"_build_qmix_observation: agent={agent.role}, pos={(x, y)}, obs={obs}"
        )

        return obs
    # -----------------------------------------------------
    # OBSERVATION für A2 + A3 (regelbasiert: einfach percepts)
    # -----------------------------------------------------
    def _build_patch_observation(self, agent):
        # A2/A3 sind regelbasiert, brauchen nur percepts
        percepts = self.world.get_percepts(agent)
        return percepts

    # -----------------------------------------------------
    # Scheduler führt ein Step aus
    # -----------------------------------------------------
    def step(self):
        if not any(getattr(a, "agent_alive", True) for a in self.agents):
            logger.info("step: ALL_DEAD detected")
            return "ALL_DEAD"

        start = self.turn
        while not getattr(self.agents[self.turn], "agent_alive", True):
            self.turn = (self.turn + 1) % len(self.agents)
            if self.turn == start:
                logger.info("step: no alive agents in rotation")
                return "ALL_DEAD"

        agent = self.agents[self.turn]

        # Agent Nachrichten erhalten (nur einmalig!)
        if self.bus:
            msgs = self.bus.get_messages_for(agent.role)
            if msgs and hasattr(agent, "receive_messages"):
                agent.receive_messages(msgs)
                logger.debug(f"step: agent={agent.role}, received {len(msgs)} message(s)")

        logger.debug(
            f"step: TURN={self.turn}, agent={agent.role}, pos={agent.pos()}, "
            f"alive={getattr(agent, 'agent_alive', True)}"
        )

        # Knowledge an A1 übergeben
        if agent.role == "A1" and hasattr(agent, "set_memory"):
            agent.set_memory(
                known=self.known,
                safe=self.safe,
                risky=self.risky,
                grid_size=self.world.grid_size,
            )

        if not getattr(agent, "agent_alive", True):
            self.turn = (self.turn + 1) % len(self.agents)
            logger.debug("step: agent not alive -> CONTINUE")
            return "CONTINUE"

        percepts = self.world.get_percepts(agent)

        # OBSERVATION bauen
        if agent.role == "A1":
            observation = self._build_qmix_observation(agent, percepts)
        else:
            observation = self._build_patch_observation(agent)

        # ACTION erfragen
        try:
            action = agent.decide_move(observation, self.world.grid_size)
            logger.debug(
                f"step: agent={agent.role}, decision_input=observation, action={action}"
            )
        except TypeError:
            action = agent.decide_move(percepts, self.world.grid_size)
            logger.debug(
                f"step: agent={agent.role}, decision_input=percepts, action={action}"
            )

        old_pos = agent.pos()
        self._update_memory_before_move(agent, percepts)

        # APPLY MOVE
        result = self.world.execute(agent, action)
        new_pos = agent.pos()
        new_percepts = self.world.get_percepts(agent)

        if agent.role == "A2" and hasattr(agent, "update_component_qs"):
            rewards = {
                "gold": 100 if new_percepts.get("glitter", 0) == 1 else 0,
                "pit": -100 if result == "DIED" and new_pos in self.world.pits else 0,
                "wumpus": -100 if result == "DIED" and new_pos in self.world.wumpus else 0,
            }

            old_state = old_pos
            new_state = new_pos
            agent.update_component_qs(old_state, action, new_state, rewards)

        logger.debug(
            f"step: agent={agent.role}, result={result}, old_pos={old_pos}, "
            f"new_pos={new_pos}, new_percepts={new_percepts}"
        )

        # Reward Shaping
        self._apply_reward_shaping(agent, old_pos, new_pos, percepts, new_percepts)

        # Memory AFTER Move
        self._update_memory_after_move(agent, new_percepts)

        # falle gestorben
        if result == "DIED":
            dead_pos = new_pos
            is_pit = dead_pos in self.world.pits
            is_wumpus = dead_pos in self.world.wumpus

            
            logger.info(f"Scheduler: agent died at {dead_pos} cause_pit={is_pit} cause_wumpus={is_wumpus}")

            for other in self.agents:
                
                if getattr(other, "role", None) == "A2":
                   
                    if is_pit and hasattr(other, "confirmed_pits"):
                        other.confirmed_pits.add(dead_pos)
                        logger.info(f"[A2] marked confirmed PIT at {dead_pos}")
                    elif is_wumpus and hasattr(other, "confirmed_wumpus"):
                        other.confirmed_wumpus.add(dead_pos)
                        logger.info(f"[A2] marked confirmed WUMPUS at {dead_pos}")
                    else:
                        
                        if hasattr(other, "confirmed_pits"):
                            other.confirmed_pits.add(dead_pos)
                            logger.info(f"[A2] marked conservative PIT at {dead_pos}")
            
            if getattr(self, "bus", None):
                cause = "pit" if is_pit else ("wumpus" if is_wumpus else "unknown")
                try:
                    self.bus.broadcast(sender="Scheduler", topic="AGENT_DIED", payload={"pos": dead_pos, "cause": cause})
                except Exception:
                    pass

        self.turn = (self.turn + 1) % len(self.agents)
        return result
