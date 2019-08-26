---
layout: page
title: About
permalink: /about/
weight: 3
---

# **About Me**

Hey, I'm {{ site.author.name }}.
I'm interested in math and machine learning.

Currently I'm trying to maximize this:
<br>

<!-- Prepare a container for your calendar. -->
<script
  src="https://cdn.rawgit.com/IonicaBizau/github-calendar/gh-pages/dist/github-calendar.min.js"
>
</script>

<!-- Optionally, include the theme (if you don't want to struggle to write the CSS) -->
<link
  rel="stylesheet"
  href="https://cdn.rawgit.com/IonicaBizau/github-calendar/gh-pages/dist/github-calendar.css"
/>

<!-- Prepare a container for your calendar. -->
<div class="calendar">
    <!-- Loading stuff -->
    Loading Github contribution data.
</div>

<script>
    new GitHubCalendar(".calendar", "your-username", {
                        responsive: true,
                        proxy: function (url) {
                        return "https://the-proxy-domain.com/req?method=GET&url=https://github.com/nunoskew";
                        }});
    new GitHubCalendar(".calendar", "nunoskew", { responsive: true });
</script>

<br>
If you like the content, feel free to drop me a message through Twitter using the small icon below.
