import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Task 1: Load and inspect the dataset
# ---------------------------------------------------------------
print("TASK 1: LOAD AND INSPECT")
df = pd.read_csv("15_cinema_sales.csv")
print("Rows and columns:", df.shape)
print(df.head())
print(df.info())
print(df.describe())

# ---------------------------------------------------------------
# Task 2: Missing values and duplicate Transaction_IDs
# ---------------------------------------------------------------
print("\nTASK 2: MISSING VALUES AND DUPLICATES")
print("Missing values in each column:")
print(df.isnull().sum())

duplicates = df["Transaction_ID"].duplicated().sum()
print("Duplicate Transaction_IDs:", duplicates)

# Keep the first copy of each Transaction_ID and drop the rest
df = df.drop_duplicates(subset="Transaction_ID")
print("Rows after removing duplicates:", len(df))

# Attach rate on the real (non-missing) data only - used in Task 10
real_spend = df["Concession_Spend"].dropna()
attach_rate_real = (real_spend > 0).sum() / len(real_spend) * 100

# ---------------------------------------------------------------
# Task 3: Fill missing Concession_Spend
# ---------------------------------------------------------------
# Spend is skewed (many zeros and a few big spenders), so the mean
# would be pulled up. The median is safer. We use the median of
# each genre, because genres have different snack habits.
print("\nTASK 3: FILL MISSING CONCESSION_SPEND")
genre_median = df.groupby("Genre")["Concession_Spend"].transform("median")
df["Concession_Spend"] = df["Concession_Spend"].fillna(genre_median)
print("Missing Concession_Spend left:", df["Concession_Spend"].isnull().sum())

# ---------------------------------------------------------------
# Task 4: Convert Show_Date and find weekday / weekend
# ---------------------------------------------------------------
print("\nTASK 4: WEEKDAY VS WEEKEND")
df["Show_Date"] = pd.to_datetime(df["Show_Date"])
df["Day_Name"] = df["Show_Date"].dt.day_name()
# dayofweek: Monday = 0 ... Saturday = 5, Sunday = 6
df["Day_Type"] = np.where(df["Show_Date"].dt.dayofweek >= 5, "Weekend", "Weekday")
print(df[["Show_Date", "Day_Name", "Day_Type"]].head())
print(df["Day_Type"].value_counts())

# ---------------------------------------------------------------
# Task 5: Revenue totals
# ---------------------------------------------------------------
print("\nTASK 5: REVENUE")
df["Ticket_Revenue"] = df["Ticket_Quantity"] * df["Ticket_Price"]
df["Total_Spend"] = df["Ticket_Revenue"] + df["Concession_Spend"]

total_ticket = df["Ticket_Revenue"].sum()
total_concession = df["Concession_Spend"].sum()
print("Total ticket revenue:", total_ticket)
print("Total concession revenue:", total_concession)
print("Combined revenue:", total_ticket + total_concession)

# ---------------------------------------------------------------
# Task 6: Top 10 movies
# ---------------------------------------------------------------
print("\nTASK 6: TOP 10 MOVIES")
movies = df.groupby("Movie_Title")[["Ticket_Quantity", "Ticket_Revenue"]].sum()

print("Top 10 by tickets sold:")
print(movies.sort_values("Ticket_Quantity", ascending=False).head(10))

print("Top 10 by ticket revenue:")
print(movies.sort_values("Ticket_Revenue", ascending=False).head(10))

# ---------------------------------------------------------------
# Task 7: Occupancy by screen format
# ---------------------------------------------------------------
print("\nTASK 7: OCCUPANCY BY SCREEN FORMAT")
format_summary = df.groupby("Screen_Format").mean(numeric_only=True)
print(format_summary[["Occupancy_Pct", "Ticket_Price", "Concession_Spend"]])

# ---------------------------------------------------------------
# Task 8: Weekday vs weekend sales
# ---------------------------------------------------------------
print("\nTASK 8: WEEKDAY VS WEEKEND")
day_type_totals = df.groupby("Day_Type")[["Ticket_Quantity", "Concession_Spend"]].sum()
print("Totals:")
print(day_type_totals)

# There are more weekdays than weekend days, so also compare per day
days_count = df.groupby("Day_Type")["Show_Date"].nunique()
print("Average tickets sold per day:")
print(day_type_totals["Ticket_Quantity"] / days_count)

print("Average concession spend per transaction:")
print(df.groupby("Day_Type")["Concession_Spend"].mean())

# ---------------------------------------------------------------
# Task 9: Attendance by showtime slot
# ---------------------------------------------------------------
print("\nTASK 9: SHOWTIME SLOTS")
slot_tickets = df.groupby("Showtime_Slot")["Ticket_Quantity"].sum()
print("Tickets sold:")
print(slot_tickets)
print("Share of all tickets (%):")
print(slot_tickets / slot_tickets.sum() * 100)
print("Average occupancy (%):")
print(df.groupby("Showtime_Slot")["Occupancy_Pct"].mean())

# ---------------------------------------------------------------
# Task 10: Concession attach rate
# ---------------------------------------------------------------
print("\nTASK 10: CONCESSION ATTACH RATE")
# Attach rate = % of transactions where snack spend is more than 0
df["Has_Concession"] = df["Concession_Spend"] > 0
attach_rate = df["Has_Concession"].mean() * 100
print("Attach rate (all rows, after filling):", round(attach_rate, 1), "%")
print("Attach rate (real data only, before filling):", round(attach_rate_real, 1), "%")
print("Attach rate by day type (%):")
print(df.groupby("Day_Type")["Has_Concession"].mean() * 100)

# ---------------------------------------------------------------
# Task 11: Ticket vs snack spending by genre
# ---------------------------------------------------------------
print("\nTASK 11: GENRE")
genre_summary = df.groupby("Genre")[["Ticket_Price", "Ticket_Revenue", "Concession_Spend"]].mean()
print(genre_summary.sort_values("Concession_Spend", ascending=False))

# ---------------------------------------------------------------
# Task 12: NumPy statistics on total transaction spend
# ---------------------------------------------------------------
print("\nTASK 12: NUMPY STATISTICS")
spend = df["Total_Spend"].to_numpy()
print("Mean:", np.mean(spend))
print("Median:", np.median(spend))
print("Standard deviation:", np.std(spend))

# ---------------------------------------------------------------
# Task 13: Large group bookings using IQR
# ---------------------------------------------------------------
print("\nTASK 13: LARGE GROUP BOOKINGS (IQR)")
qty = df["Ticket_Quantity"].to_numpy()
q1 = np.percentile(qty, 25)
q3 = np.percentile(qty, 75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
print("Q1:", q1, " Q3:", q3, " IQR:", iqr)
print("Upper limit:", upper_limit)

large_groups = df[df["Ticket_Quantity"] > upper_limit]
print("Number of unusually large bookings:", len(large_groups))
print(large_groups[["Transaction_ID", "Movie_Title", "Ticket_Quantity", "Total_Spend"]].head(10))

# ---------------------------------------------------------------
# Task 14: Ticket quantity bands
# ---------------------------------------------------------------
print("\nTASK 14: TICKET QUANTITY BANDS")
bins = [0, 1, 2, 5, 100]
names = ["Solo", "Couple", "Small Group", "Large Group"]
df["Group_Band"] = pd.cut(df["Ticket_Quantity"], bins=bins, labels=names)
print(df["Group_Band"].value_counts())
print("Average snack and total spend by band:")
print(df.groupby("Group_Band", observed=True)[["Concession_Spend", "Total_Spend"]].mean())

# ---------------------------------------------------------------
# Task 15: Recommendations (numbers come from the results above)
# ---------------------------------------------------------------
print("\nTASK 15: RECOMMENDATIONS")

best_format = format_summary["Occupancy_Pct"].idxmax()
worst_format = format_summary["Occupancy_Pct"].idxmin()
best_slot = slot_tickets.idxmax()
slot_occupancy = df.groupby("Showtime_Slot")["Occupancy_Pct"].mean()
weakest_slot = slot_occupancy.idxmin()
top_movie = movies["Ticket_Quantity"].idxmax()
snack_genre = genre_summary["Concession_Spend"].idxmax()
low_snack_genre = genre_summary["Concession_Spend"].idxmin()
weekend_per_day = day_type_totals.loc["Weekend", "Ticket_Quantity"] / days_count["Weekend"]
weekday_per_day = day_type_totals.loc["Weekday", "Ticket_Quantity"] / days_count["Weekday"]

print("1.", best_format, "has the highest occupancy (",
      round(format_summary.loc[best_format, "Occupancy_Pct"], 1), "%) vs",
      worst_format, "(", round(format_summary.loc[worst_format, "Occupancy_Pct"], 1),
      "%). Give more screens to", best_format, "for popular movies like", top_movie, ".")

print("2. The", best_slot, "slot sells the most tickets, and", weakest_slot,
      "has the lowest occupancy (", round(slot_occupancy[weakest_slot], 1),
      "%). Add combo discounts or cut screenings in the", weakest_slot, "slot.")

print("3. Weekends sell", round(weekend_per_day, 1), "tickets per day vs",
      round(weekday_per_day, 1), "on weekdays. Run weekend combo promotions and",
      "weekday offers to fill empty seats.")

print("4.", snack_genre, "movies have the highest snack spend (",
      round(genre_summary.loc[snack_genre, "Concession_Spend"], 1), ") while",
      low_snack_genre, "has the lowest (",
      round(genre_summary.loc[low_snack_genre, "Concession_Spend"], 1),
      "). Offer family combos for", snack_genre, "and pre-order snack deals for",
      low_snack_genre, ".")

print("5.", len(large_groups), "bookings are unusually large groups (more than",
      upper_limit, "tickets). Create a special group package with bundled snacks.")

# ---------------------------------------------------------------
# Save results as CSV files (same folder as this script)
# ---------------------------------------------------------------
df.to_csv("output_cleaned_data.csv", index=False)
movies.sort_values("Ticket_Quantity", ascending=False).head(10).to_csv("output_top_movies.csv")
format_summary.to_csv("output_screen_format.csv")
genre_summary.to_csv("output_genre.csv")
large_groups.to_csv("output_large_groups.csv", index=False)
print("\nSaved: output_cleaned_data.csv, output_top_movies.csv, output_screen_format.csv,")
print("       output_genre.csv, output_large_groups.csv")
