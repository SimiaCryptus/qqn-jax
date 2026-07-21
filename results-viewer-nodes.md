Need to be able to filter the list based on:
* activation type
* time bounds (datetime selector)

The currently selected filters and the currently selected optimizer items should be persisted in the url for linkability and persistence. When loading, the first matching item by name in the filtered list is used (there may be duplicates, though the filter is intended to disambiguate)
When the filter is changed, e.g. the activation function, the optimizer items selected should attempt a re-resolution using the same names. this is to support each navigation between the equivalent plots when varying e.g. activation function


Provide a selectable color scheme/theme

Also, when selecting or mousing over data points in the graph, it should show all the data about that point, not just the plotted coordinates. if we are plotting vs time, we want the details to still show evaluation count, for example
