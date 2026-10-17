import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';
import { LoginPageComponent } from './pages/login-page/login-page.component';
import { CreatePageComponent } from './pages/create-page/create-page.component';
import { SetBuilderPageComponent } from './pages/set-builder-page/set-builder-page.component';
import { GalleryPageComponent } from './pages/gallery-page/gallery-page.component';
import { CommunityPageComponent } from './pages/community-page/community-page.component';
import { VotePageComponent } from './pages/vote-page/vote-page.component';
import { EventHistoryPageComponent } from './pages/event-history-page/event-history-page.component';
import { AdminPageComponent } from './pages/admin-page/admin-page.component';
import { PoolPageComponent } from './pages/pool-page/pool-page.component';
import { TokenMakerPageComponent } from './pages/token-maker-page/token-maker-page.component';
import { ResultsPageComponent } from './pages/results-page/results-page.component';

/** Spec §7 routes. Everything except /login needs a stored user. */
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'create' },
  { path: 'login', component: LoginPageComponent, title: 'Log in · MTGenesis.AI' },
  { path: 'create', component: CreatePageComponent, canActivate: [authGuard], title: 'Create · MTGenesis.AI' },
  { path: 'tokens', component: TokenMakerPageComponent, canActivate: [authGuard], title: 'Token Maker · MTGenesis.AI' },
  { path: 'set', component: SetBuilderPageComponent, canActivate: [authGuard], title: 'Commanders · MTGenesis.AI' },
  { path: 'gallery', component: GalleryPageComponent, canActivate: [authGuard], title: 'My Gallery · MTGenesis.AI' },
  { path: 'community', component: CommunityPageComponent, canActivate: [authGuard], title: 'Community Gallery · MTGenesis.AI' },
  { path: 'vote', component: VotePageComponent, canActivate: [authGuard], title: 'Vote · MTGenesis.AI' },
  { path: 'events', component: EventHistoryPageComponent, canActivate: [authGuard], title: 'Past Events · MTGenesis.AI' },
  { path: 'events/:id', component: EventHistoryPageComponent, canActivate: [authGuard], title: 'Event · MTGenesis.AI' },
  { path: 'pool', component: PoolPageComponent, canActivate: [authGuard], title: 'Knowledge Pool · MTGenesis.AI' },
  { path: 'pool/:id', component: PoolPageComponent, canActivate: [authGuard], title: 'Knowledge Pool · MTGenesis.AI' },
  { path: 'results', component: ResultsPageComponent, canActivate: [authGuard], title: 'Results · MTGenesis.AI' },
  { path: 'admin', component: AdminPageComponent, canActivate: [authGuard], title: 'Host · MTGenesis.AI' },
  { path: '**', redirectTo: 'create' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
