import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { authGuard } from './guards/auth.guard';
import { LoginPageComponent } from './pages/login-page/login-page.component';
import { CreatePageComponent } from './pages/create-page/create-page.component';
import { SetBuilderPageComponent } from './pages/set-builder-page/set-builder-page.component';
import { GalleryPageComponent } from './pages/gallery-page/gallery-page.component';
import { VotePageComponent } from './pages/vote-page/vote-page.component';
import { EventHistoryPageComponent } from './pages/event-history-page/event-history-page.component';
import { AdminPageComponent } from './pages/admin-page/admin-page.component';

/** Spec §7 routes. Everything except /login needs a stored user. */
export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'create' },
  { path: 'login', component: LoginPageComponent, title: 'Log in · MTGenesis.AI' },
  { path: 'create', component: CreatePageComponent, canActivate: [authGuard], title: 'Create · MTGenesis.AI' },
  { path: 'set', component: SetBuilderPageComponent, canActivate: [authGuard], title: 'Commander Set · MTGenesis.AI' },
  { path: 'gallery', component: GalleryPageComponent, canActivate: [authGuard], title: 'Gallery · MTGenesis.AI' },
  { path: 'vote', component: VotePageComponent, canActivate: [authGuard], title: 'Vote · MTGenesis.AI' },
  { path: 'events', component: EventHistoryPageComponent, canActivate: [authGuard], title: 'Past Events · MTGenesis.AI' },
  { path: 'events/:id', component: EventHistoryPageComponent, canActivate: [authGuard], title: 'Event · MTGenesis.AI' },
  { path: 'admin', component: AdminPageComponent, canActivate: [authGuard], title: 'Host · MTGenesis.AI' },
  { path: '**', redirectTo: 'create' }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
