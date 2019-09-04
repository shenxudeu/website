---
layout:     post
title:      "Information Theory of Quantitative Trading"
subtitle:   "A design pattern of trading system"
date:       2019-09-01
author:     "Shen Xu"
image: "img/generate_digit_map_2.png"
published: true
hide-in-home: false
markup: "mmark"
tags:
    - Finance
categories: [ Finance ]
---


Information Theory of Quantitative Trading
==================================

I have two theories to support myself when designing a trading system, mechanism theory and information theory. Mechanism theory is straight forward, more or less, we express our understanding of market mechanism by designing a trading system. For example, a statistical arbitrage system reflects designer's (or trader's) view of market order book structures. An event trader build strategy upon her understanding of market participants in those events. Without market mechanism understanding , it's difficult to build a "robust" system purely from historical market data.

Here, I want to talk about a slightly different view, information theory. Market is called "efficient" if it reflects all available related information. My information theory emphaphize on bringing information into market to make it efficient.

Let's look at quantitative trading history from information view first.

#### Digitizing of market data

The digitizing of market price volume information created the first wave of quantitative trading. Once quants are equipped with price data and a computer, he can start to explore market inefficiencies. Those famous factors (eg. momentum etc) and risk premium had been discovered quickly. This leads to numerous CTA strategies and funds. CTA funds automated the work of traditional chart traders and gradually replaced them. AQR and Winton are great examples.

#### Speed of communication

Starting from 2005, thanks to the development of high speed networks, quants start to shift their attentions to bringing first hand information to market. Market rewards those flash boys by their market making serves. The speed competetion leads to automation of traditional market makers and replaced them in past decade. By now, not only US stock market, but also Futures markets are mostly made by Chicago high frequency firm like Jump trading, Cetaldel and etc.

#### Complex information beyond price

In recent years, an area called alternative data trading attracts a lot attentions. Financial text news, social media contents, credit card transaction information are all under this alternative data umbrella. The AI booming also helps the development of processing alternative data. These programs have also been enjoying compensation by providing market information. The only difference is, this time, they try to replace traditional market analyst.

From this short history, a patten can be seen, technique development leads to processing more complicated information with higher speed. Market, itself as an information exchange, rewards people providing information service.

So, I always think my profession as a strategist, is an information server. What I can be rewarded, if any, should come from the quality of my service. This also serves as one of my trading strategy design pattens.
