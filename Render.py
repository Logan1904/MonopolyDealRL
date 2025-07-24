from rich.console import Console
from rich.table import Table
from rich.text import Text

from mappings import *
from Card import *

class Render():
    def __init__(self):
        self.console = Console()
    
    def render(self, mode, internal_state):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        players, agents, agent_selection, deck, action_context = internal_state

        player = players[agent_selection]

        if mode == 'pre':
            print("-"*75 + str(agent_selection) + "-"*75)
            print("")
            print("Deck Size: " + str(deck.deckSize()))
            print("Discard Size: " + str(deck.discardSize()))
            print("")
            self.render_list(player.hand, label="Hand")
            self.render_list(player.money, label="Money")
            print("Properties: ")
            self.render_properties(player.sets)
            print("")
        elif mode == 'post':
            print("")
            print("Deck Size: " + str(deck.deckSize()))
            print("Discard Size: " + str(deck.discardSize()))
            print("")
            self.render_list(player.hand, label="Hand")
            self.render_list(player.money, label="Money")
            print("Properties: ")
            self.render_properties(player.sets)
            print("")
            print("")
        elif mode == 'action':
            self.render_action(agents, action_context)

    def render_properties(self, sets):
        console = Console()
        table = Table(show_lines=True)

        table.add_column("PropertySet")
        for colour in sets.keys():
            table.add_column(colour)

        num_pSets = len(next(iter(sets.values())))
        for i in range(num_pSets):
            row = [f"[{i}]"]
            for colour in sets:
                pSet = sets[colour][i]
                if pSet.isEmpty():
                    row.append(Text("--", style="dim"))
                else:
                    cell_text = Text()
                    for card in pSet.properties:
                        cell_text.append(f"{card.name}\n", style=COLOUR_STYLE_MAP[colour])  # newline for vertical stack
                    row.append(cell_text)
                    
            table.add_row(*row)

        console.print(table)

    def render_list(self, cards, label):
        console = Console()

        # Sort cards
        sorted_cards = sorted(cards, key=self.get_card_sort_key)

        # Build styled line
        line = Text(f"{label}: ")
        for card in sorted_cards:
            style = self.get_card_style(card)
            line.append(f"[{card.name}] ", style=style)

        console.print(line)

    def render_action(self, agents, action_context):
        console = Console()

        table = Table(title="Action", show_lines=True)
        table.add_column("Field", style="bold cyan", justify="right")
        table.add_column("Value")

        # Always show action
        action = action_context["action"]
        table.add_row("Action:", f"{ACTION_DESCRIPTION[action]}")

        if action == 0:
            pass
        elif action == 1:
            table.add_row("Card:", f"{CARD_MAPPING[action_context["my_property"]["card"]]}")
            table.add_row("Moved to Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("Moved to Set:", str(action_context["my_set"]["set_index"]))
        elif action == 2:
            card_ID = action_context["hand_card"]
            table.add_row("Card:", f"{CARD_MAPPING[card_ID]}")
        elif action in [3,4]:
            card_ID = action_context["hand_card"]
            table.add_row("Card:", f"{CARD_MAPPING[card_ID]}")
            table.add_row("Played to Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("Played to Set:", str(action_context["my_set"]["set_index"]))
        elif action == 5:
            table.add_row("Opponent:", f"{agents[action_context["opponent_ID"]]}")
            card_ID = action_context["opponent_property"]["card"]
            table.add_row("Stealing:", f"{CARD_MAPPING[card_ID]}")
            table.add_row("Played to Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("Played to Set:", str(action_context["my_set"]["set_index"]))
        elif action == 6:
            table.add_row("Opponent:", f"{agents[action_context["opponent_ID"]]}")
            card_ID = action_context["my_property"]["card"]
            table.add_row("Swapping:", f"{CARD_MAPPING[card_ID]}")
            card_ID = action_context["opponent_property"]["card"]
            table.add_row("For:", f"{CARD_MAPPING[card_ID]}")
            table.add_row("Played to Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("Played to Set:", str(action_context["my_set"]["set_index"]))
        elif action in [7,8]:
            table.add_row("Opponent:", f"{agents[action_context["opponent_ID"]]}")
        elif action == 9:
            table.add_row("Opponent:", f"{agents[action_context["opponent_ID"]]}")
            table.add_row("Stealing Colour:", action_context["opponent_set"]["colour"])
            table.add_row("Stealing Set:", action_context["opponent_set"]["set_index"])
        elif action in [10,11,12,13,14]:
            table.add_row("On Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("On Set:", str(action_context["my_set"]["set_index"]))
        elif action == 15:
            table.add_row("Opponent:", f"{agents[action_context["opponent_ID"]]}")
            table.add_row("On Colour:", COLOUR_MAPPING[action_context["my_set"]["colour"]])
            table.add_row("On Set:", str(action_context["my_set"]["set_index"]))
        elif action in [16]:
            pass
        
        console.print(table)

    def get_card_style(self, card):
        if isinstance(card, MoneyCard):
            return "green"
        elif isinstance(card, RentCard):
            return "white"
        elif isinstance(card, ActionCard):
            return "red"
        elif isinstance(card, PropertyCard):
            return COLOUR_STYLE_MAP[card.colours[0]]
        
    def get_card_sort_key(self, card):
        if isinstance(card, MoneyCard):
            return 0
        elif isinstance(card, RentCard):
            return 1
        elif isinstance(card, ActionCard):
            return 2
        elif isinstance(card, PropertyCard):
            return 3